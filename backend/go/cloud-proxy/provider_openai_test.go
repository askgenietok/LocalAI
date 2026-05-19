package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	pb "github.com/mudler/LocalAI/pkg/grpc/proto"
)

// fakeOpenAIUpstream returns an httptest.Server that decodes the
// inbound request as an openAIRequest, calls handler with it, and
// writes the handler's reply as the response.
func fakeOpenAIUpstream(t *testing.T, handler func(req openAIRequest) (status int, body string, contentType string)) (*httptest.Server, *openAIRequest) {
	t.Helper()
	var captured openAIRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &captured)
		status, body, ct := handler(captured)
		w.Header().Set("Content-Type", ct)
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	return srv, &captured
}

func newTranslateCloudProxy(t *testing.T, upstreamURL string) *CloudProxy {
	t.Helper()
	os.Setenv("CLOUD_PROXY_OPENAI_FAKE", "sk-fake-openai")
	t.Cleanup(func() { os.Unsetenv("CLOUD_PROXY_OPENAI_FAKE") })
	cp := NewCloudProxy()
	if err := cp.Load(&pb.ModelOptions{
		Model: "gpt-4o-local",
		Proxy: &pb.ProxyOptions{
			UpstreamUrl:   upstreamURL,
			Mode:          modeTranslate,
			Provider:      providerOpenAI,
			ApiKeyEnv:     "CLOUD_PROXY_OPENAI_FAKE",
			UpstreamModel: "gpt-4o",
		},
	}); err != nil {
		t.Fatal(err)
	}
	return cp
}

func TestPredict_OpenAI_BasicChat(t *testing.T) {
	srv, captured := fakeOpenAIUpstream(t, func(_ openAIRequest) (int, string, string) {
		return 200, `{"id":"resp-1","choices":[{"index":0,"message":{"role":"assistant","content":"hi there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`, "application/json"
	})
	defer srv.Close()
	cp := newTranslateCloudProxy(t, srv.URL)

	got, err := cp.Predict(&pb.PredictOptions{
		Messages: []*pb.Message{
			{Role: "system", Content: "be brief"},
			{Role: "user", Content: "hello"},
		},
		Temperature: 0.5,
		TopP:        0.9,
		Tokens:      32,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != "hi there" {
		t.Fatalf("got %q", got)
	}

	// Verify the upstream saw a properly-translated request.
	if captured.Model != "gpt-4o" {
		t.Fatalf("upstream model=%q want gpt-4o (from upstream_model override)", captured.Model)
	}
	if len(captured.Messages) != 2 {
		t.Fatalf("upstream got %d messages, want 2", len(captured.Messages))
	}
	if captured.Messages[0].Role != "system" || captured.Messages[1].Role != "user" {
		t.Fatalf("upstream messages=%+v", captured.Messages)
	}
	if captured.Temperature == nil || *captured.Temperature != 0.5 {
		t.Fatalf("temperature not forwarded: %+v", captured.Temperature)
	}
	if captured.MaxTokens == nil || *captured.MaxTokens != 32 {
		t.Fatalf("max_tokens not forwarded: %+v", captured.MaxTokens)
	}
	if captured.Stream {
		t.Fatal("expected non-streaming request")
	}
}

func TestPredict_OpenAI_PromptFallback(t *testing.T) {
	// No Messages array — backend should synth a single user message
	// from Prompt so non-chat clients still route through translate.
	srv, captured := fakeOpenAIUpstream(t, func(_ openAIRequest) (int, string, string) {
		return 200, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`, "application/json"
	})
	defer srv.Close()
	cp := newTranslateCloudProxy(t, srv.URL)

	_, err := cp.Predict(&pb.PredictOptions{Prompt: "what time is it?"})
	if err != nil {
		t.Fatal(err)
	}
	if len(captured.Messages) != 1 || captured.Messages[0].Role != "user" || captured.Messages[0].Content != "what time is it?" {
		t.Fatalf("prompt fallback failed: %+v", captured.Messages)
	}
}

func TestPredict_OpenAI_UpstreamError(t *testing.T) {
	srv, _ := fakeOpenAIUpstream(t, func(_ openAIRequest) (int, string, string) {
		return 401, `{"error":{"message":"bad key"}}`, "application/json"
	})
	defer srv.Close()
	cp := newTranslateCloudProxy(t, srv.URL)

	_, err := cp.Predict(&pb.PredictOptions{Messages: []*pb.Message{{Role: "user", Content: "x"}}})
	if err == nil || !strings.Contains(err.Error(), "401") {
		t.Fatalf("expected 401 error, got %v", err)
	}
}

func TestPredictStream_OpenAI_StreamsContent(t *testing.T) {
	// Stream three content deltas then [DONE]. Verify the channel
	// receives them in order with no missing pieces.
	chunks := []string{
		`{"choices":[{"index":0,"delta":{"role":"assistant"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"hello"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":" "}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"world"}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}
	body := ""
	for _, c := range chunks {
		body += "data: " + c + "\n\n"
	}
	body += "data: [DONE]\n\n"

	srv, captured := fakeOpenAIUpstream(t, func(_ openAIRequest) (int, string, string) {
		return 200, body, "text/event-stream"
	})
	defer srv.Close()
	cp := newTranslateCloudProxy(t, srv.URL)

	results := make(chan string, 8)
	done := make(chan error, 1)
	go func() {
		done <- cp.PredictStream(&pb.PredictOptions{
			Messages: []*pb.Message{{Role: "user", Content: "hi"}},
		}, results)
	}()

	var got []string
	for s := range results {
		got = append(got, s)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, "") != "hello world" {
		t.Fatalf("got %q", got)
	}
	if !captured.Stream {
		t.Fatal("upstream did not see stream:true")
	}
}

func TestPredict_RejectedInPassthroughMode(t *testing.T) {
	os.Setenv("CLOUD_PROXY_FAKE", "k")
	t.Cleanup(func() { os.Unsetenv("CLOUD_PROXY_FAKE") })
	cp := NewCloudProxy()
	if err := cp.Load(&pb.ModelOptions{
		Proxy: &pb.ProxyOptions{
			UpstreamUrl: "https://example.com",
			Mode:        modePassthrough,
			ApiKeyEnv:   "CLOUD_PROXY_FAKE",
		},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := cp.Predict(&pb.PredictOptions{}); err == nil || !strings.Contains(err.Error(), "only valid in translate") {
		t.Fatalf("expected mode-mismatch error, got %v", err)
	}
}

// unused-import linting guard
var _ = fmt.Sprintf

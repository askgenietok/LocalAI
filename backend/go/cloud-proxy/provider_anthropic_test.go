package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	pb "github.com/mudler/LocalAI/pkg/grpc/proto"
)

// fakeAnthropicUpstream mirrors fakeOpenAIUpstream but decodes the
// request body as an anthropicRequest so tests can assert on the
// translated wire shape (system field, max_tokens, etc.).
func fakeAnthropicUpstream(t *testing.T, handler func(req anthropicRequest) (status int, body string, contentType string)) (*httptest.Server, *anthropicRequest) {
	t.Helper()
	var captured anthropicRequest
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

func newAnthropicTranslateCloudProxy(t *testing.T, upstreamURL string) *CloudProxy {
	t.Helper()
	os.Setenv("CLOUD_PROXY_ANTHROPIC_FAKE", "sk-ant-fake")
	t.Cleanup(func() { os.Unsetenv("CLOUD_PROXY_ANTHROPIC_FAKE") })
	cp := NewCloudProxy()
	if err := cp.Load(&pb.ModelOptions{
		Model: "claude-local",
		Proxy: &pb.ProxyOptions{
			UpstreamUrl:   upstreamURL,
			Mode:          modeTranslate,
			Provider:      providerAnthropic,
			ApiKeyEnv:     "CLOUD_PROXY_ANTHROPIC_FAKE",
			UpstreamModel: "claude-3-5-sonnet-20241022",
		},
	}); err != nil {
		t.Fatal(err)
	}
	return cp
}

func TestPredict_Anthropic_BasicMessages(t *testing.T) {
	srv, captured := fakeAnthropicUpstream(t, func(_ anthropicRequest) (int, string, string) {
		return 200, `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hi there"}],"model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":5,"output_tokens":2}}`, "application/json"
	})
	defer srv.Close()
	cp := newAnthropicTranslateCloudProxy(t, srv.URL)

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

	if captured.Model != "claude-3-5-sonnet-20241022" {
		t.Fatalf("upstream model=%q", captured.Model)
	}
	// System message must be hoisted out of Messages into top-level field.
	if captured.System != "be brief" {
		t.Fatalf("system=%q want %q", captured.System, "be brief")
	}
	if len(captured.Messages) != 1 || captured.Messages[0].Role != "user" {
		t.Fatalf("messages=%+v", captured.Messages)
	}
	if captured.MaxTokens != 32 {
		t.Fatalf("max_tokens=%d want 32", captured.MaxTokens)
	}
	if captured.Temperature == nil || *captured.Temperature != 0.5 {
		t.Fatalf("temperature not forwarded: %+v", captured.Temperature)
	}
	if captured.Stream {
		t.Fatal("expected non-streaming request")
	}
}

func TestPredict_Anthropic_DefaultsMaxTokens(t *testing.T) {
	// Anthropic 400s without max_tokens. The translator must default
	// it when the caller doesn't supply Tokens.
	srv, captured := fakeAnthropicUpstream(t, func(_ anthropicRequest) (int, string, string) {
		return 200, `{"content":[{"type":"text","text":"ok"}]}`, "application/json"
	})
	defer srv.Close()
	cp := newAnthropicTranslateCloudProxy(t, srv.URL)

	_, err := cp.Predict(&pb.PredictOptions{Messages: []*pb.Message{{Role: "user", Content: "x"}}})
	if err != nil {
		t.Fatal(err)
	}
	if captured.MaxTokens != anthropicDefaultMaxTokens {
		t.Fatalf("max_tokens=%d want default %d", captured.MaxTokens, anthropicDefaultMaxTokens)
	}
}

func TestPredict_Anthropic_PromptFallback(t *testing.T) {
	srv, captured := fakeAnthropicUpstream(t, func(_ anthropicRequest) (int, string, string) {
		return 200, `{"content":[{"type":"text","text":"ok"}]}`, "application/json"
	})
	defer srv.Close()
	cp := newAnthropicTranslateCloudProxy(t, srv.URL)

	_, err := cp.Predict(&pb.PredictOptions{Prompt: "what time is it?", Tokens: 16})
	if err != nil {
		t.Fatal(err)
	}
	if len(captured.Messages) != 1 || captured.Messages[0].Role != "user" || captured.Messages[0].Content != "what time is it?" {
		t.Fatalf("prompt fallback failed: %+v", captured.Messages)
	}
}

func TestPredict_Anthropic_ConcatenatesContentBlocks(t *testing.T) {
	// Anthropic may return multiple text blocks; the translator joins
	// them so the Predict() string return is the full assistant message.
	srv, _ := fakeAnthropicUpstream(t, func(_ anthropicRequest) (int, string, string) {
		return 200, `{"content":[{"type":"text","text":"hello "},{"type":"text","text":"world"}]}`, "application/json"
	})
	defer srv.Close()
	cp := newAnthropicTranslateCloudProxy(t, srv.URL)

	got, err := cp.Predict(&pb.PredictOptions{Messages: []*pb.Message{{Role: "user", Content: "x"}}, Tokens: 16})
	if err != nil {
		t.Fatal(err)
	}
	if got != "hello world" {
		t.Fatalf("got %q", got)
	}
}

func TestPredict_Anthropic_UpstreamError(t *testing.T) {
	srv, _ := fakeAnthropicUpstream(t, func(_ anthropicRequest) (int, string, string) {
		return 401, `{"error":{"type":"authentication_error","message":"bad key"}}`, "application/json"
	})
	defer srv.Close()
	cp := newAnthropicTranslateCloudProxy(t, srv.URL)

	_, err := cp.Predict(&pb.PredictOptions{Messages: []*pb.Message{{Role: "user", Content: "x"}}, Tokens: 16})
	if err == nil || !strings.Contains(err.Error(), "401") {
		t.Fatalf("expected 401 error, got %v", err)
	}
}

func TestPredictStream_Anthropic_StreamsTextDeltas(t *testing.T) {
	// Real Anthropic SSE has event: lines + data: lines. The translator
	// only needs the data: payload; only content_block_delta with
	// delta.type=text_delta carries content. message_stop ends.
	frames := []string{
		"event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" \"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"world\"}}\n\n",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	}
	body := strings.Join(frames, "")

	srv, captured := fakeAnthropicUpstream(t, func(_ anthropicRequest) (int, string, string) {
		return 200, body, "text/event-stream"
	})
	defer srv.Close()
	cp := newAnthropicTranslateCloudProxy(t, srv.URL)

	results := make(chan string, 8)
	done := make(chan error, 1)
	go func() {
		done <- cp.PredictStream(&pb.PredictOptions{
			Messages: []*pb.Message{{Role: "user", Content: "hi"}},
			Tokens:   16,
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

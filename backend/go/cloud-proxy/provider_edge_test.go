package main

import (
	"encoding/json"
	"strings"
	"testing"

	pb "github.com/mudler/LocalAI/pkg/grpc/proto"
)

// Verify buildOpenAIRequest preserves caller-supplied tools and
// tool_choice as opaque JSON. PredictOptions carries them as strings;
// they must land in the outbound request body unchanged so the
// upstream sees the caller's intent verbatim. A regression here would
// silently disable function calling for translate-mode clients.
func TestBuildOpenAIRequest_ToolsAndToolChoicePassthrough(t *testing.T) {
	cfg := &proxyConfig{upstreamModel: "gpt-4o"}
	toolsJSON := `[{"type":"function","function":{"name":"search","parameters":{"type":"object"}}}]`
	choiceJSON := `{"type":"function","function":{"name":"search"}}`

	body, err := buildOpenAIRequest(&pb.PredictOptions{
		Messages:   []*pb.Message{{Role: "user", Content: "find x"}},
		Tools:      toolsJSON,
		ToolChoice: choiceJSON,
	}, cfg, false)
	if err != nil {
		t.Fatal(err)
	}

	var decoded openAIRequest
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatal(err)
	}
	// Compare the JSON-canonical form so whitespace differences are ignored.
	gotTools, _ := json.Marshal(json.RawMessage(decoded.Tools))
	wantTools, _ := json.Marshal(json.RawMessage(toolsJSON))
	if string(gotTools) != string(wantTools) {
		t.Fatalf("tools mismatch: got %s want %s", gotTools, wantTools)
	}
	gotChoice, _ := json.Marshal(json.RawMessage(decoded.ToolChoice))
	wantChoice, _ := json.Marshal(json.RawMessage(choiceJSON))
	if string(gotChoice) != string(wantChoice) {
		t.Fatalf("tool_choice mismatch: got %s want %s", gotChoice, wantChoice)
	}
}

// Garbage JSON in tools / tool_choice is silently dropped (omitted)
// rather than blowing up the request. Documents the parseRawJSON
// behaviour — operators shouldn't see hard failures from an upstream
// caller's mis-formatted tools field.
func TestBuildOpenAIRequest_InvalidToolsJSONDropped(t *testing.T) {
	cfg := &proxyConfig{upstreamModel: "gpt-4o"}
	body, err := buildOpenAIRequest(&pb.PredictOptions{
		Messages:   []*pb.Message{{Role: "user", Content: "x"}},
		Tools:      "this is not json",
		ToolChoice: "{also bad",
	}, cfg, false)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), "this is not json") {
		t.Fatalf("invalid tools leaked into body: %s", body)
	}
	if strings.Contains(string(body), "{also bad") {
		t.Fatalf("invalid tool_choice leaked into body: %s", body)
	}
}

// Anthropic empty content array yields an empty Reply (not an error).
// Mirrors how an upstream tool_use-only response might arrive — the
// content array can legitimately be empty in some edge cases.
func TestPredictRich_Anthropic_EmptyContent(t *testing.T) {
	srv, _ := fakeAnthropicUpstream(t, func(_ anthropicRequest) (int, string, string) {
		return 200, `{"id":"m1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":3,"output_tokens":0}}`, "application/json"
	})
	defer srv.Close()
	cp := newAnthropicTranslateCloudProxy(t, srv.URL)

	reply, err := cp.PredictRich(&pb.PredictOptions{
		Messages: []*pb.Message{{Role: "user", Content: "x"}},
		Tokens:   16,
	})
	if err != nil {
		t.Fatalf("empty content should not error: %v", err)
	}
	if string(reply.GetMessage()) != "" {
		t.Fatalf("expected empty message, got %q", reply.GetMessage())
	}
	if len(reply.GetChatDeltas()) != 0 {
		t.Fatalf("expected no chat deltas, got %d", len(reply.GetChatDeltas()))
	}
	if reply.GetPromptTokens() != 3 {
		t.Fatalf("usage tokens not propagated: %d", reply.GetPromptTokens())
	}
}

// A truncated / malformed SSE payload mid-stream should be tolerated:
// the malformed chunk gets skipped (xlog.Debug logged), valid chunks
// before AND after it still reach the channel.
func TestPredictStreamRich_OpenAI_TolerantOfBadChunks(t *testing.T) {
	body := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}`,
		``,
		`data: this-is-not-json{{`,
		``,
		`data: {"choices":[{"index":0,"delta":{"content":" world"}}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	srv, _ := fakeOpenAIUpstream(t, func(_ openAIRequest) (int, string, string) {
		return 200, body, "text/event-stream"
	})
	defer srv.Close()
	cp := newTranslateCloudProxy(t, srv.URL)

	results := make(chan *pb.Reply, 8)
	done := make(chan error, 1)
	go func() {
		done <- cp.PredictStreamRich(&pb.PredictOptions{
			Messages: []*pb.Message{{Role: "user", Content: "hi"}},
		}, results)
		close(results)
	}()

	var assembled strings.Builder
	for reply := range results {
		assembled.Write(reply.GetMessage())
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	// The good chunks before and after the malformed one both made it through.
	if assembled.String() != "hello world" {
		t.Fatalf("got %q want %q", assembled.String(), "hello world")
	}
}

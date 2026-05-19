package main

import (
	"strings"
	"testing"

	pb "github.com/mudler/LocalAI/pkg/grpc/proto"
)

// OpenAI: non-streaming tool call response. Verify the response is
// mapped to Reply.ChatDeltas[].ToolCalls with id/name/arguments intact,
// and usage tokens land on Reply.PromptTokens / Reply.Tokens.
func TestPredictRich_OpenAI_ToolCalls(t *testing.T) {
	srv, _ := fakeOpenAIUpstream(t, func(_ openAIRequest) (int, string, string) {
		return 200, `{
			"id":"resp-1",
			"choices":[{
				"index":0,
				"message":{
					"role":"assistant",
					"content":"",
					"tool_calls":[
						{"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"SF\"}"}},
						{"id":"call_def","type":"function","function":{"name":"get_time","arguments":"{\"tz\":\"PT\"}"}}
					]
				},
				"finish_reason":"tool_calls"
			}],
			"usage":{"prompt_tokens":42,"completion_tokens":18,"total_tokens":60}
		}`, "application/json"
	})
	defer srv.Close()
	cp := newTranslateCloudProxy(t, srv.URL)

	reply, err := cp.PredictRich(&pb.PredictOptions{
		Messages: []*pb.Message{{Role: "user", Content: "what's the weather?"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if string(reply.GetMessage()) != "" {
		t.Fatalf("expected empty content with tool_calls, got %q", reply.GetMessage())
	}
	if reply.GetPromptTokens() != 42 || reply.GetTokens() != 18 {
		t.Fatalf("usage tokens not propagated: prompt=%d completion=%d", reply.GetPromptTokens(), reply.GetTokens())
	}
	if len(reply.GetChatDeltas()) != 1 {
		t.Fatalf("expected 1 ChatDelta, got %d", len(reply.GetChatDeltas()))
	}
	tcs := reply.GetChatDeltas()[0].GetToolCalls()
	if len(tcs) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(tcs))
	}
	if tcs[0].GetId() != "call_abc" || tcs[0].GetName() != "get_weather" {
		t.Fatalf("tool call 0 wrong: %+v", tcs[0])
	}
	if !strings.Contains(tcs[0].GetArguments(), `"location":"SF"`) {
		t.Fatalf("tool call 0 args missing payload: %q", tcs[0].GetArguments())
	}
	if tcs[1].GetId() != "call_def" || tcs[1].GetName() != "get_time" {
		t.Fatalf("tool call 1 wrong: %+v", tcs[1])
	}
}

// OpenAI: streaming tool call. Arguments arrive as a sequence of
// delta chunks; the consumer is expected to concatenate by tool index.
// Verify each chunk reaches the channel and the assembled arguments
// match the input.
func TestPredictStreamRich_OpenAI_ToolCallDeltas(t *testing.T) {
	chunks := []string{
		// Frame 0: announce the tool call (id + name, no args yet).
		`{"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"search"}}]}}]}`,
		// Frames 1-3: arguments arrive in fragments.
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"clo"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"uds\"}"}}]}}]}`,
		// Stop frame.
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	}
	body := ""
	for _, c := range chunks {
		body += "data: " + c + "\n\n"
	}
	body += "data: [DONE]\n\n"

	srv, _ := fakeOpenAIUpstream(t, func(_ openAIRequest) (int, string, string) {
		return 200, body, "text/event-stream"
	})
	defer srv.Close()
	cp := newTranslateCloudProxy(t, srv.URL)

	results := make(chan *pb.Reply, 16)
	done := make(chan error, 1)
	go func() {
		done <- cp.PredictStreamRich(&pb.PredictOptions{
			Messages: []*pb.Message{{Role: "user", Content: "find something"}},
		}, results)
		close(results)
	}()

	var (
		toolName  string
		toolID    string
		toolIndex int32 = -1
		argsBuf   strings.Builder
	)
	for reply := range results {
		for _, cd := range reply.GetChatDeltas() {
			for _, tc := range cd.GetToolCalls() {
				if tc.GetName() != "" {
					toolName = tc.GetName()
				}
				if tc.GetId() != "" {
					toolID = tc.GetId()
				}
				if toolIndex == -1 {
					toolIndex = tc.GetIndex()
				}
				argsBuf.WriteString(tc.GetArguments())
			}
		}
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if toolID != "call_xyz" || toolName != "search" {
		t.Fatalf("tool call header lost: id=%q name=%q", toolID, toolName)
	}
	if toolIndex != 0 {
		t.Fatalf("tool index=%d want 0", toolIndex)
	}
	if argsBuf.String() != `{"q":"clouds"}` {
		t.Fatalf("assembled args=%q want %q", argsBuf.String(), `{"q":"clouds"}`)
	}
}

// Anthropic: non-streaming tool_use block. The block appears in
// Content[] alongside text blocks; the input field is a structured
// JSON object. Map to ToolCallDelta with arguments as serialised JSON
// so downstream OpenAI-shaped consumers see a familiar format.
func TestPredictRich_Anthropic_ToolUse(t *testing.T) {
	srv, _ := fakeAnthropicUpstream(t, func(_ anthropicRequest) (int, string, string) {
		return 200, `{
			"id":"msg_1","type":"message","role":"assistant",
			"content":[
				{"type":"text","text":"Let me check that."},
				{"type":"tool_use","id":"toolu_01","name":"weather","input":{"location":"SF"}}
			],
			"model":"claude","usage":{"input_tokens":12,"output_tokens":34}
		}`, "application/json"
	})
	defer srv.Close()
	cp := newAnthropicTranslateCloudProxy(t, srv.URL)

	reply, err := cp.PredictRich(&pb.PredictOptions{
		Messages: []*pb.Message{{Role: "user", Content: "what's the weather?"}},
		Tokens:   64,
	})
	if err != nil {
		t.Fatal(err)
	}
	if string(reply.GetMessage()) != "Let me check that." {
		t.Fatalf("content=%q", reply.GetMessage())
	}
	if reply.GetPromptTokens() != 12 || reply.GetTokens() != 34 {
		t.Fatalf("usage tokens: prompt=%d completion=%d", reply.GetPromptTokens(), reply.GetTokens())
	}
	if len(reply.GetChatDeltas()) != 1 || len(reply.GetChatDeltas()[0].GetToolCalls()) != 1 {
		t.Fatalf("expected 1 tool call, got %+v", reply.GetChatDeltas())
	}
	tc := reply.GetChatDeltas()[0].GetToolCalls()[0]
	if tc.GetId() != "toolu_01" || tc.GetName() != "weather" {
		t.Fatalf("tool call wrong: %+v", tc)
	}
	if !strings.Contains(tc.GetArguments(), `"location":"SF"`) {
		t.Fatalf("tool call args missing payload: %q", tc.GetArguments())
	}
}

// Anthropic: streaming tool_use. content_block_start announces the
// tool's id + name; input_json_delta events carry argument fragments
// which the consumer accumulates. message_delta carries final usage.
func TestPredictStreamRich_Anthropic_InputJSONDelta(t *testing.T) {
	frames := []string{
		"event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
		// Block 0 is a tool_use; consumer should allocate a slot.
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_42\",\"name\":\"lookup\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"rain\\\"}\"}}\n\n",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":7}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	}
	body := strings.Join(frames, "")

	srv, _ := fakeAnthropicUpstream(t, func(_ anthropicRequest) (int, string, string) {
		return 200, body, "text/event-stream"
	})
	defer srv.Close()
	cp := newAnthropicTranslateCloudProxy(t, srv.URL)

	results := make(chan *pb.Reply, 16)
	done := make(chan error, 1)
	go func() {
		done <- cp.PredictStreamRich(&pb.PredictOptions{
			Messages: []*pb.Message{{Role: "user", Content: "rain?"}},
			Tokens:   64,
		}, results)
		close(results)
	}()

	var (
		toolID, toolName string
		argsBuf          strings.Builder
		finalTokens      int32
	)
	for reply := range results {
		if reply.GetTokens() > 0 && len(reply.GetChatDeltas()) == 0 {
			finalTokens = reply.GetTokens()
			continue
		}
		for _, cd := range reply.GetChatDeltas() {
			for _, tc := range cd.GetToolCalls() {
				if tc.GetId() != "" {
					toolID = tc.GetId()
				}
				if tc.GetName() != "" {
					toolName = tc.GetName()
				}
				argsBuf.WriteString(tc.GetArguments())
			}
		}
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if toolID != "toolu_42" || toolName != "lookup" {
		t.Fatalf("tool init lost: id=%q name=%q", toolID, toolName)
	}
	if argsBuf.String() != `{"q":"rain"}` {
		t.Fatalf("assembled args=%q want %q", argsBuf.String(), `{"q":"rain"}`)
	}
	if finalTokens != 7 {
		t.Fatalf("final usage tokens=%d want 7", finalTokens)
	}
}

// Sanity: the legacy Predict() (string, error) signature still works
// — it delegates to PredictRich and extracts Message.
func TestPredict_LegacyWrapper_OpenAI(t *testing.T) {
	srv, _ := fakeOpenAIUpstream(t, func(_ openAIRequest) (int, string, string) {
		return 200, `{"choices":[{"message":{"role":"assistant","content":"hello"}}]}`, "application/json"
	})
	defer srv.Close()
	cp := newTranslateCloudProxy(t, srv.URL)

	got, err := cp.Predict(&pb.PredictOptions{Messages: []*pb.Message{{Role: "user", Content: "hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	if got != "hello" {
		t.Fatalf("got %q", got)
	}
}

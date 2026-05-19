package main

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	grpc "github.com/mudler/LocalAI/pkg/grpc"
	pb "github.com/mudler/LocalAI/pkg/grpc/proto"
)

// helper: run a CloudProxy in-process via grpc.Provide so tests can
// call Forward through the public Backend interface without listening
// on a real socket.
func newInProcClient(t *testing.T, proxy *CloudProxy) grpc.Backend {
	t.Helper()
	addr := "test://" + t.Name()
	grpc.Provide(addr, proxy)
	return grpc.NewClient(addr, true, nil, false)
}

func TestForward_PassthroughEcho(t *testing.T) {
	// Fake upstream: echoes the request body back, prefixed with a
	// canary so the test can assert both that the body reached the
	// upstream and the response made it back to the client.
	gotBody := make(chan string, 1)
	gotAuth := make(chan string, 1)
	gotPath := make(chan string, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		gotBody <- string(body)
		gotAuth <- r.Header.Get("Authorization")
		gotPath <- r.URL.Path
		w.Header().Set("X-Echo", "true")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("echo: " + string(body)))
	}))
	defer upstream.Close()

	os.Setenv("CLOUD_PROXY_FAKE_KEY", "sk-fake")
	defer os.Unsetenv("CLOUD_PROXY_FAKE_KEY")

	cp := NewCloudProxy()
	if err := cp.Load(&pb.ModelOptions{
		Proxy: &pb.ProxyOptions{
			UpstreamUrl: upstream.URL,
			Mode:        modePassthrough,
			ApiKeyEnv:   "CLOUD_PROXY_FAKE_KEY",
		},
	}); err != nil {
		t.Fatal(err)
	}

	c := newInProcClient(t, cp)
	stream, err := c.Forward(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	if err := stream.Send(&pb.ForwardRequest{
		Path:      "/v1/chat/completions",
		Method:    "POST",
		Headers:   []*pb.ForwardHeader{{Name: "Content-Type", Value: "application/json"}},
		BodyChunk: []byte(`{"prompt":`),
	}); err != nil {
		t.Fatal(err)
	}
	if err := stream.Send(&pb.ForwardRequest{BodyChunk: []byte(`"hi"}`)}); err != nil {
		t.Fatal(err)
	}
	if err := stream.CloseSend(); err != nil {
		t.Fatal(err)
	}

	// First reply: status + headers.
	first, err := stream.Recv()
	if err != nil {
		t.Fatal(err)
	}
	if first.Status != http.StatusOK {
		t.Fatalf("status=%d want 200", first.Status)
	}
	if !hasHeader(first.Headers, "X-Echo", "true") {
		t.Fatalf("missing X-Echo header in reply: %v", first.Headers)
	}

	// Subsequent replies: body.
	var body []byte
	for {
		r, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		body = append(body, r.BodyChunk...)
	}
	if string(body) != `echo: {"prompt":"hi"}` {
		t.Fatalf("body=%q", string(body))
	}

	// Upstream observations.
	select {
	case got := <-gotBody:
		if got != `{"prompt":"hi"}` {
			t.Fatalf("upstream got body=%q", got)
		}
	case <-time.After(time.Second):
		t.Fatal("upstream never saw body")
	}
	select {
	case got := <-gotAuth:
		if got != "Bearer sk-fake" {
			t.Fatalf("upstream auth=%q want Bearer sk-fake", got)
		}
	case <-time.After(time.Second):
		t.Fatal("upstream never saw auth header")
	}
	select {
	case got := <-gotPath:
		if got != "/v1/chat/completions" {
			t.Fatalf("upstream path=%q", got)
		}
	case <-time.After(time.Second):
		t.Fatal("upstream never saw path")
	}
}

func TestForward_AnthropicAuthHeader(t *testing.T) {
	gotXAPIKey := make(chan string, 1)
	gotVersion := make(chan string, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotXAPIKey <- r.Header.Get("x-api-key")
		gotVersion <- r.Header.Get("anthropic-version")
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	os.Setenv("CLOUD_PROXY_ANTHROPIC_KEY", "sk-ant-fake")
	defer os.Unsetenv("CLOUD_PROXY_ANTHROPIC_KEY")

	cp := NewCloudProxy()
	if err := cp.Load(&pb.ModelOptions{
		Proxy: &pb.ProxyOptions{
			UpstreamUrl: upstream.URL,
			Mode:        modePassthrough,
			Provider:    providerAnthropic,
			ApiKeyEnv:   "CLOUD_PROXY_ANTHROPIC_KEY",
		},
	}); err != nil {
		t.Fatal(err)
	}

	c := newInProcClient(t, cp)
	stream, err := c.Forward(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := stream.Send(&pb.ForwardRequest{Path: "/v1/messages", Method: "POST"}); err != nil {
		t.Fatal(err)
	}
	_ = stream.CloseSend()
	_, _ = stream.Recv() // drain status
	for {
		if _, err := stream.Recv(); errors.Is(err, io.EOF) || err != nil {
			break
		}
	}

	if got := <-gotXAPIKey; got != "sk-ant-fake" {
		t.Fatalf("x-api-key=%q", got)
	}
	if got := <-gotVersion; got == "" {
		t.Fatal("anthropic-version not set")
	}
}

func TestLoad_ValidatesConfig(t *testing.T) {
	cp := NewCloudProxy()

	if err := cp.Load(&pb.ModelOptions{}); err == nil || !strings.Contains(err.Error(), "ProxyOptions") {
		t.Fatalf("expected missing ProxyOptions error, got %v", err)
	}

	if err := cp.Load(&pb.ModelOptions{Proxy: &pb.ProxyOptions{}}); err == nil || !strings.Contains(err.Error(), "upstream_url") {
		t.Fatalf("expected missing upstream_url error, got %v", err)
	}

	if err := cp.Load(&pb.ModelOptions{Proxy: &pb.ProxyOptions{
		UpstreamUrl: "https://example.com",
		Mode:        "rewrite",
	}}); err == nil || !strings.Contains(err.Error(), "unknown mode") {
		t.Fatalf("expected unknown-mode error, got %v", err)
	}

	// translate + openai should load successfully (Phase 5).
	if err := cp.Load(&pb.ModelOptions{Proxy: &pb.ProxyOptions{
		UpstreamUrl: "https://example.com/v1/chat/completions",
		Mode:        modeTranslate,
		Provider:    providerOpenAI,
	}}); err != nil {
		t.Fatalf("expected translate+openai to load, got %v", err)
	}

	// translate + anthropic should load successfully (Phase 6).
	if err := cp.Load(&pb.ModelOptions{Proxy: &pb.ProxyOptions{
		UpstreamUrl: "https://example.com/v1/messages",
		Mode:        modeTranslate,
		Provider:    providerAnthropic,
	}}); err != nil {
		t.Fatalf("expected translate+anthropic to load, got %v", err)
	}

	if err := cp.Load(&pb.ModelOptions{Proxy: &pb.ProxyOptions{
		UpstreamUrl: "https://example.com",
		ApiKeyEnv:   "DEFINITELY_UNSET_ENV_VAR_XYZ",
	}}); err == nil || !strings.Contains(err.Error(), "unset") {
		t.Fatalf("expected unset-env error, got %v", err)
	}
}

func TestForward_RejectsWithoutLoad(t *testing.T) {
	cp := NewCloudProxy()
	c := newInProcClient(t, cp)
	stream, err := c.Forward(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	_ = stream.CloseSend()
	if _, err := stream.Recv(); err == nil || !strings.Contains(err.Error(), "not loaded") {
		t.Fatalf("expected not-loaded error, got %v", err)
	}
}

func hasHeader(hs []*pb.ForwardHeader, name, value string) bool {
	for _, h := range hs {
		if strings.EqualFold(h.GetName(), name) && h.GetValue() == value {
			return true
		}
	}
	return false
}

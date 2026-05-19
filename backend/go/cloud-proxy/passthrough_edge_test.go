package main

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"sync"
	"testing"

	grpc "github.com/mudler/LocalAI/pkg/grpc"
	pb "github.com/mudler/LocalAI/pkg/grpc/proto"
)

func TestComposeURL(t *testing.T) {
	// Upstream URL convention: gallery configs put the canonical path
	// in upstream_url, so per-request Path is ignored. A bare-host
	// upstream_url accepts the per-request path. Verify both branches.
	cases := []struct {
		name     string
		upstream string
		reqPath  string
		want     string
	}{
		{"full path wins", "https://api.openai.com/v1/chat/completions", "/v1/something-else", "https://api.openai.com/v1/chat/completions"},
		{"bare host accepts path", "https://api.openai.com", "/v1/chat/completions", "https://api.openai.com/v1/chat/completions"},
		{"root slash treated as bare", "https://api.openai.com/", "/v1/chat/completions", "https://api.openai.com/v1/chat/completions"},
		{"bare host + empty path", "https://api.openai.com", "", "https://api.openai.com"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := composeURL(tc.upstream, tc.reqPath)
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestComposeURL_InvalidUpstream(t *testing.T) {
	_, err := composeURL("://garbage", "")
	if err == nil {
		t.Fatal("expected error on invalid upstream URL")
	}
}

func TestApplyAuthHeader_AnthropicSetsXAPIKeyAndVersion(t *testing.T) {
	req, _ := http.NewRequest("POST", "https://example.com", nil)
	applyAuthHeader(req, providerAnthropic, "ant-key")
	if got := req.Header.Get("x-api-key"); got != "ant-key" {
		t.Fatalf("x-api-key=%q", got)
	}
	if req.Header.Get("anthropic-version") == "" {
		t.Fatal("anthropic-version not set")
	}
	if req.Header.Get("Authorization") != "" {
		t.Fatal("Authorization leaked on anthropic backend")
	}
}

func TestApplyAuthHeader_OpenAISetsBearer(t *testing.T) {
	req, _ := http.NewRequest("POST", "https://example.com", nil)
	applyAuthHeader(req, providerOpenAI, "sk-key")
	if got := req.Header.Get("Authorization"); got != "Bearer sk-key" {
		t.Fatalf("Authorization=%q", got)
	}
	if req.Header.Get("x-api-key") != "" {
		t.Fatal("x-api-key leaked on openai backend")
	}
}

func TestApplyAuthHeader_EmptyProviderDefaultsBearer(t *testing.T) {
	// Passthrough mode often has provider == "" because the operator
	// doesn't claim a specific upstream wire format. Most providers
	// (including OpenAI-compatible ones) accept Bearer, so default to it.
	req, _ := http.NewRequest("POST", "https://example.com", nil)
	applyAuthHeader(req, "", "some-key")
	if got := req.Header.Get("Authorization"); got != "Bearer some-key" {
		t.Fatalf("Authorization=%q", got)
	}
}

func TestApplyAuthHeader_AnthropicPreservesExistingVersion(t *testing.T) {
	// If the client supplied anthropic-version (rare but legitimate
	// for an upstream that's pinned to a specific date), the proxy
	// must not clobber it.
	req, _ := http.NewRequest("POST", "https://example.com", nil)
	req.Header.Set("anthropic-version", "2024-10-01")
	applyAuthHeader(req, providerAnthropic, "k")
	if got := req.Header.Get("anthropic-version"); got != "2024-10-01" {
		t.Fatalf("anthropic-version clobbered: %q", got)
	}
}

func TestIsHopByHopHeader(t *testing.T) {
	cases := map[string]bool{
		"Connection":        true,
		"Keep-Alive":        true,
		"Proxy-Connection":  true,
		"Transfer-Encoding": true,
		"TE":                true,
		"Trailer":           true,
		"Upgrade":           true,
		"Host":              true,
		"Content-Length":    true,
		// Case-insensitive — RFC 7230 doesn't constrain header case.
		"connection": true,
		"HOST":       true,
		// Non hop-by-hop — must NOT be stripped.
		"Authorization": false,
		"Content-Type":  false,
		"Accept":        false,
		"X-Custom":      false,
	}
	for h, want := range cases {
		if got := isHopByHopHeader(h); got != want {
			t.Errorf("isHopByHopHeader(%q)=%v want %v", h, got, want)
		}
	}
}

func TestForward_StripsHopByHopAndConnectionHeadersBeforeUpstream(t *testing.T) {
	// Caller sends Connection / Transfer-Encoding; verify they don't
	// reach the upstream where they'd confuse the HTTP client (or in
	// rare cases leak a connection-management hint that doesn't belong
	// across the proxy boundary).
	gotConnection := make(chan string, 1)
	gotXCustom := make(chan string, 1)
	gotHost := make(chan string, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotConnection <- r.Header.Get("Connection")
		gotXCustom <- r.Header.Get("X-Custom")
		gotHost <- r.Header.Get("Host")
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	cp := NewCloudProxy()
	if err := cp.Load(&pb.ModelOptions{
		Proxy: &pb.ProxyOptions{
			UpstreamUrl: upstream.URL,
			Mode:        modePassthrough,
		},
	}); err != nil {
		t.Fatal(err)
	}

	addr := "test://forward-hopbyhop"
	grpc.Provide(addr, cp)
	c := grpc.NewClient(addr, true, nil, false)
	stream, err := c.Forward(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := stream.Send(&pb.ForwardRequest{
		Path:   "/v1/chat/completions",
		Method: "POST",
		Headers: []*pb.ForwardHeader{
			{Name: "Connection", Value: "keep-alive"},
			{Name: "Host", Value: "spoofed.example.com"},
			{Name: "X-Custom", Value: "preserved"},
		},
	}); err != nil {
		t.Fatal(err)
	}
	_ = stream.CloseSend()
	_, _ = stream.Recv()
	for {
		if _, err := stream.Recv(); errors.Is(err, io.EOF) || err != nil {
			break
		}
	}

	if got := <-gotConnection; got != "" {
		t.Errorf("Connection leaked to upstream: %q", got)
	}
	if got := <-gotHost; got == "spoofed.example.com" {
		t.Errorf("Host header spoofed through to upstream: %q", got)
	}
	if got := <-gotXCustom; got != "preserved" {
		t.Errorf("X-Custom header was stripped: %q", got)
	}
}

func TestForward_ReplacesCallerSuppliedAuthorization(t *testing.T) {
	// The proxy must overwrite a client-supplied Authorization header so
	// a downstream caller can't smuggle stale or wrong credentials.
	gotAuth := make(chan string, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth <- r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
	}))
	defer upstream.Close()

	os.Setenv("CLOUD_PROXY_AUTH_REPLACE_KEY", "sk-real")
	defer os.Unsetenv("CLOUD_PROXY_AUTH_REPLACE_KEY")

	cp := NewCloudProxy()
	if err := cp.Load(&pb.ModelOptions{
		Proxy: &pb.ProxyOptions{
			UpstreamUrl: upstream.URL,
			Mode:        modePassthrough,
			ApiKeyEnv:   "CLOUD_PROXY_AUTH_REPLACE_KEY",
		},
	}); err != nil {
		t.Fatal(err)
	}

	addr := "test://forward-replaces-auth"
	grpc.Provide(addr, cp)
	c := grpc.NewClient(addr, true, nil, false)
	stream, _ := c.Forward(context.Background())
	if err := stream.Send(&pb.ForwardRequest{
		Path:   "/v1/chat/completions",
		Method: "POST",
		Headers: []*pb.ForwardHeader{
			// Client-supplied Authorization with the wrong scheme / wrong key.
			{Name: "Authorization", Value: "Basic Zm9vOmJhcg=="},
		},
	}); err != nil {
		t.Fatal(err)
	}
	_ = stream.CloseSend()
	_, _ = stream.Recv()
	for {
		if _, err := stream.Recv(); errors.Is(err, io.EOF) || err != nil {
			break
		}
	}

	got := <-gotAuth
	if got != "Bearer sk-real" {
		t.Fatalf("upstream auth=%q want Bearer sk-real (caller Basic header should be replaced)", got)
	}
}

func TestForward_ConcurrentCallsDoNotInterfere(t *testing.T) {
	// CloudProxy explicitly omits base.SingleThread — independent
	// Forward streams must not block each other or leak state across
	// requests. Drive a few in parallel and verify each gets its own
	// response.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		// Echo the body back so we can confirm each request's payload
		// rounds-trips independently.
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body)
	}))
	defer upstream.Close()

	cp := NewCloudProxy()
	if err := cp.Load(&pb.ModelOptions{
		Proxy: &pb.ProxyOptions{
			UpstreamUrl: upstream.URL,
			Mode:        modePassthrough,
		},
	}); err != nil {
		t.Fatal(err)
	}
	addr := "test://forward-concurrent"
	grpc.Provide(addr, cp)
	c := grpc.NewClient(addr, true, nil, false)

	const N = 8
	var wg sync.WaitGroup
	errs := make(chan error, N)
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			stream, err := c.Forward(context.Background())
			if err != nil {
				errs <- err
				return
			}
			payload := "request-" + string(rune('A'+idx))
			if err := stream.Send(&pb.ForwardRequest{
				Path:      "/v1/chat/completions",
				Method:    "POST",
				BodyChunk: []byte(payload),
			}); err != nil {
				errs <- err
				return
			}
			_ = stream.CloseSend()
			_, _ = stream.Recv()
			var body []byte
			for {
				r, err := stream.Recv()
				if errors.Is(err, io.EOF) {
					break
				}
				if err != nil {
					errs <- err
					return
				}
				body = append(body, r.GetBodyChunk()...)
			}
			if string(body) != payload {
				errs <- &echoMismatch{want: payload, got: string(body)}
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent Forward failed: %v", err)
	}
}

type echoMismatch struct{ want, got string }

func (e *echoMismatch) Error() string {
	return "echo mismatch: want " + strconv.Quote(e.want) + " got " + strconv.Quote(e.got)
}

package cloudproxy

import (
	"context"
	"errors"
	"io"
	"testing"
	"time"

	pb "github.com/mudler/LocalAI/pkg/grpc/proto"
)

// scriptedForwardClient is a fake ForwardClient that returns a fixed
// sequence of replies. Each Recv pops the next reply or returns the
// terminal error. Used to drive forwardReader through scripted gRPC
// responses without standing up a real backend.
type scriptedForwardClient struct {
	replies []*pb.ForwardReply
	final   error
	idx     int
}

func (s *scriptedForwardClient) Send(*pb.ForwardRequest) error { return nil }
func (s *scriptedForwardClient) CloseSend() error              { return nil }
func (s *scriptedForwardClient) Context() context.Context      { return context.Background() }
func (s *scriptedForwardClient) Recv() (*pb.ForwardReply, error) {
	if s.idx >= len(s.replies) {
		if s.final != nil {
			return nil, s.final
		}
		return nil, io.EOF
	}
	r := s.replies[s.idx]
	s.idx++
	return r, nil
}

func TestForwardReader_ConcatsChunks(t *testing.T) {
	r := &forwardReader{stream: &scriptedForwardClient{
		replies: []*pb.ForwardReply{
			{BodyChunk: []byte("hello ")},
			{BodyChunk: []byte("world")},
			{BodyChunk: []byte("!")},
		},
	}}
	got, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "hello world!" {
		t.Fatalf("got %q", got)
	}
}

func TestForwardReader_PartialReads(t *testing.T) {
	r := &forwardReader{stream: &scriptedForwardClient{
		replies: []*pb.ForwardReply{
			{BodyChunk: []byte("abcdefghij")},
		},
	}}
	// Read 3 bytes at a time — exercises pos advancement within a chunk.
	var out []byte
	buf := make([]byte, 3)
	for {
		n, err := r.Read(buf)
		out = append(out, buf[:n]...)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if string(out) != "abcdefghij" {
		t.Fatalf("got %q", out)
	}
}

func TestForwardReader_SkipsEmptyChunks(t *testing.T) {
	// Empty chunks must not be treated as EOF — backends may legitimately
	// emit them (e.g. SSE keepalives, transport quirks).
	r := &forwardReader{stream: &scriptedForwardClient{
		replies: []*pb.ForwardReply{
			{BodyChunk: nil},
			{BodyChunk: []byte("data")},
			{BodyChunk: []byte{}},
			{BodyChunk: []byte("more")},
		},
	}}
	got, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "datamore" {
		t.Fatalf("got %q", got)
	}
}

// infiniteForwardClient simulates a misbehaving backend that never
// stops emitting replies. Used to verify Close() doesn't spin forever.
type infiniteForwardClient struct {
	ctx   context.Context
	calls int
}

func (s *infiniteForwardClient) Send(*pb.ForwardRequest) error { return nil }
func (s *infiniteForwardClient) CloseSend() error              { return nil }
func (s *infiniteForwardClient) Context() context.Context {
	if s.ctx == nil {
		return context.Background()
	}
	return s.ctx
}
func (s *infiniteForwardClient) Recv() (*pb.ForwardReply, error) {
	s.calls++
	return &pb.ForwardReply{BodyChunk: []byte("never-ending")}, nil
}

func TestForwardReader_CloseBoundedByIterationCap(t *testing.T) {
	// Misbehaving backend that never returns EOF. Without the cap,
	// Close() would loop forever. The cap is currently 1024.
	upstream := &infiniteForwardClient{}
	r := &forwardReader{stream: upstream}

	done := make(chan struct{})
	go func() {
		_ = r.Close()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatalf("Close did not return within 2s; calls so far: %d", upstream.calls)
	}
	if upstream.calls > 2048 {
		t.Fatalf("Close drained %d replies; expected bounded near 1024", upstream.calls)
	}
}

func TestForwardReader_CloseExitsOnContextCancel(t *testing.T) {
	// Even before the iteration cap, a cancelled context should let
	// Close() return — that's the request-scoped exit path.
	ctx, cancel := context.WithCancel(context.Background())
	upstream := &infiniteForwardClient{ctx: ctx}
	r := &forwardReader{stream: upstream}

	cancel()
	done := make(chan struct{})
	go func() {
		_ = r.Close()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatalf("Close did not return after context cancel")
	}
}

func TestForwardReader_PropagatesError(t *testing.T) {
	wantErr := errors.New("upstream blew up")
	r := &forwardReader{stream: &scriptedForwardClient{
		replies: []*pb.ForwardReply{{BodyChunk: []byte("partial")}},
		final:   wantErr,
	}}
	buf := make([]byte, 16)
	n, err := r.Read(buf)
	if n != len("partial") || string(buf[:n]) != "partial" {
		t.Fatalf("first read got %q n=%d", buf[:n], n)
	}
	if err != nil {
		t.Fatalf("first read err=%v want nil", err)
	}
	if _, err := r.Read(buf); !errors.Is(err, wantErr) {
		t.Fatalf("second read err=%v want %v", err, wantErr)
	}
}


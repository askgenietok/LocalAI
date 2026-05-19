package main

// cloud-proxy is a LocalAI backend that forwards request traffic to an
// external HTTP provider (OpenAI, Anthropic, etc.). Two modes:
//
//   - passthrough: serves the Forward RPC; the client wire format is
//     preserved end-to-end, no translation.
//   - translate: serves Predict/PredictStream; the backend converts
//     internal proto to the provider's wire format. (Phases 5–6.)
//
// LoadModel reads UpstreamURL/Mode/Provider/key references from
// ProxyOptions and resolves the API key once at load time.

import (
	"flag"
	"os"

	grpc "github.com/mudler/LocalAI/pkg/grpc"
	"github.com/mudler/xlog"
)

var addr = flag.String("addr", "localhost:50051", "the address to listen on")

func main() {
	xlog.SetLogger(xlog.NewLogger(xlog.LogLevel(os.Getenv("LOCALAI_LOG_LEVEL")), os.Getenv("LOCALAI_LOG_FORMAT")))
	flag.Parse()
	if err := grpc.StartServer(*addr, NewCloudProxy()); err != nil {
		panic(err)
	}
}

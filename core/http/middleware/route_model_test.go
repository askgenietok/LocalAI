package middleware_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"

	"github.com/labstack/echo/v4"
	"github.com/mudler/LocalAI/core/backend"
	"github.com/mudler/LocalAI/core/config"
	. "github.com/mudler/LocalAI/core/http/middleware"
	"github.com/mudler/LocalAI/core/schema"
	"github.com/mudler/LocalAI/core/services/routing/router"
	"github.com/mudler/LocalAI/pkg/system"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gopkg.in/yaml.v3"
)

// The RouteModel middleware wires the score classifier into request
// rewriting. The classifier itself is covered in
// router/score_test.go — these specs pin the middleware-level
// behaviour: candidate matching against the active label set, the
// fallback path, and the depth-1 invariant.

var _ = Describe("RouteModel middleware (score classifier)", func() {
	var (
		modelDir  string
		appConfig *config.ApplicationConfig
		loader    *config.ModelConfigLoader
		store     *fakeDecisionStore
	)

	BeforeEach(func() {
		d, err := os.MkdirTemp("", "router-test-*")
		Expect(err).NotTo(HaveOccurred())
		modelDir = d
		appConfig = &config.ApplicationConfig{
			Context:     context.Background(),
			SystemState: &system.SystemState{Model: system.Model{ModelsPath: modelDir}},
		}
		loader = config.NewModelConfigLoader(modelDir)
		store = &fakeDecisionStore{}
	})

	AfterEach(func() {
		_ = os.RemoveAll(modelDir)
	})

	It("routes to a candidate whose labels cover the active set", func() {
		// 3 policies, 2 candidates. Small model has [casual-chat],
		// bigger has [code-generation, math-reasoning, casual-chat].
		// A query that activates code-generation should fall to the
		// bigger candidate because it's the only one that covers it.
		routerCfg := newScoreRouterModel(modelDir, "smart-router")
		writeCandidate(modelDir, "small-model")
		writeCandidate(modelDir, "big-model")

		s := &stubScorer{labelToLogProb: map[string]float64{
			"code-generation": -0.05, // dominant
			"casual-chat":     -3.0,
			"math-reasoning":  -4.0,
		}}
		rec, err := runRouter(loader, appConfig, store, routerCfg, openAIChat("debug my Go null pointer"), stubScorerFactory(s))
		Expect(err).NotTo(HaveOccurred())
		Expect(rec.Code).To(Equal(http.StatusOK))
		Expect(rec.Body.String()).To(Equal("served:big-model"))
		Expect(store.records).To(HaveLen(1))
		Expect(store.records[0].ServedModel).To(Equal("big-model"))
		Expect(store.records[0].Label).To(ContainSubstring("code-generation"))
	})

	It("prefers the smaller candidate when both cover the active set", func() {
		// Both candidates list casual-chat. Admins order small →
		// big, so a casual-chat-only request must route to small.
		routerCfg := newScoreRouterModel(modelDir, "smart-router")
		writeCandidate(modelDir, "small-model")
		writeCandidate(modelDir, "big-model")

		s := &stubScorer{labelToLogProb: map[string]float64{
			"code-generation": -5.0,
			"casual-chat":     -0.05, // dominant
			"math-reasoning":  -5.0,
		}}
		rec, err := runRouter(loader, appConfig, store, routerCfg, openAIChat("hi"), stubScorerFactory(s))
		Expect(err).NotTo(HaveOccurred())
		Expect(rec.Body.String()).To(Equal("served:small-model"))
	})

	It("falls back when no candidate covers the active label set", func() {
		// Only the bigger candidate covers math-reasoning. We
		// deliberately drop it from the candidates list so neither
		// matches; expect Fallback to fire.
		routerCfg := newScoreRouterModel(modelDir, "smart-router")
		// Remove the second candidate so coverage gap appears.
		routerCfg.Router.Candidates = routerCfg.Router.Candidates[:1]
		writeCandidate(modelDir, "small-model")
		writeCandidate(modelDir, "qwen3-0.6b")

		s := &stubScorer{labelToLogProb: map[string]float64{
			"code-generation": -5.0,
			"casual-chat":     -5.0,
			"math-reasoning":  -0.05, // dominant — but no candidate has it
		}}
		rec, err := runRouter(loader, appConfig, store, routerCfg, openAIChat("3 apples cost $2.40"), stubScorerFactory(s))
		Expect(err).NotTo(HaveOccurred())
		Expect(rec.Body.String()).To(Equal("served:qwen3-0.6b"))
	})

	It("rejects candidates that reference unknown labels at build time", func() {
		routerCfg := newScoreRouterModel(modelDir, "smart-router")
		routerCfg.Router.Candidates = append(routerCfg.Router.Candidates, config.RouterCandidate{
			Model:  "broken",
			Labels: []string{"nonexistent-label"},
		})
		writeCandidate(modelDir, "small-model")
		writeCandidate(modelDir, "big-model")
		writeCandidate(modelDir, "broken")
		writeCandidate(modelDir, "qwen3-0.6b")

		s := &stubScorer{labelToLogProb: map[string]float64{
			"code-generation": -0.05,
			"casual-chat":     -3.0,
			"math-reasoning":  -4.0,
		}}
		rec, err := runRouter(loader, appConfig, store, routerCfg, openAIChat("debug something"), stubScorerFactory(s))
		// Unknown-label config bug surfaces via the
		// classifier-unavailable path, which falls through to the
		// configured Fallback.
		Expect(err).NotTo(HaveOccurred())
		Expect(rec.Body.String()).To(Equal("served:qwen3-0.6b"))
	})

	It("returns 500 when the candidate is itself a router (depth-1 invariant)", func() {
		// The candidate model is itself a router. We must reject
		// the dispatch — chained routers are deliberately
		// disallowed.
		routerCfg := newScoreRouterModel(modelDir, "smart-router")
		// Bend the test setup: replace one of the candidate-model
		// configs with a nested-router config.
		nestedRouter := newScoreRouterModel(modelDir, "small-model")
		Expect(os.WriteFile(filepath.Join(modelDir, "small-model.yaml"), []byte(toYAML(nestedRouter)), 0o644)).To(Succeed())
		writeCandidate(modelDir, "big-model")
		writeCandidate(modelDir, "qwen3-0.6b")

		s := &stubScorer{labelToLogProb: map[string]float64{
			"code-generation": -5.0,
			"casual-chat":     -0.05,
			"math-reasoning":  -5.0,
		}}
		_, err := runRouter(loader, appConfig, store, routerCfg, openAIChat("hi"), stubScorerFactory(s))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("depth-1 invariant"))
	})
})

// --- helpers ---

// stubScorer scores each candidate label according to a fixed
// label→log-prob map; per-token length is faked at 2 tokens so length
// normalisation is a no-op.
type stubScorer struct {
	labelToLogProb map[string]float64
}

func (s *stubScorer) Score(_ context.Context, _ string, candidates []string) ([]backend.CandidateScore, error) {
	out := make([]backend.CandidateScore, len(candidates))
	for i, c := range candidates {
		lp := s.labelToLogProb[c]
		out[i] = backend.CandidateScore{
			LogProb:                 lp * 2,
			LengthNormalizedLogProb: lp,
			NumTokens:               2,
		}
	}
	return out, nil
}

func stubScorerFactory(s *stubScorer) ScorerFactory {
	return func(string) backend.Scorer { return s }
}

type fakeDecisionStore struct {
	records []router.DecisionRecord
}

func (f *fakeDecisionStore) Record(_ context.Context, r router.DecisionRecord) error {
	f.records = append(f.records, r)
	return nil
}

func (f *fakeDecisionStore) List(_ context.Context, _ router.DecisionListQuery) ([]router.DecisionRecord, error) {
	out := append([]router.DecisionRecord(nil), f.records...)
	return out, nil
}

func (f *fakeDecisionStore) Close() error                         { return nil }
func (f *fakeDecisionStore) Count(_ context.Context) (int, error) { return len(f.records), nil }

// newScoreRouterModel builds a smart-router config with 3 policies
// and 2 candidates (small with one label, bigger with all three).
// Admins are expected to order candidates small → large; the
// middleware picks the first whose labels are a superset of the
// active set.
func newScoreRouterModel(modelDir, name string) *config.ModelConfig {
	cfg := &config.ModelConfig{
		Name: name,
		Router: config.RouterConfig{
			Classifier:      "score",
			ClassifierModel: "arch-router",
			Fallback:        "qwen3-0.6b",
			Policies: []config.RouterPolicy{
				{Label: "code-generation", Description: "writing or debugging code"},
				{Label: "casual-chat", Description: "small talk"},
				{Label: "math-reasoning", Description: "arithmetic and word problems"},
			},
			Candidates: []config.RouterCandidate{
				{Model: "small-model", Labels: []string{"casual-chat"}},
				{Model: "big-model", Labels: []string{"code-generation", "casual-chat", "math-reasoning"}},
			},
		},
	}
	Expect(os.WriteFile(filepath.Join(modelDir, name+".yaml"), []byte(toYAML(cfg)), 0o644)).To(Succeed())
	return cfg
}

func writeCandidate(modelDir, name string) {
	body := "name: " + name + "\nbackend: mock-backend\n"
	Expect(os.WriteFile(filepath.Join(modelDir, name+".yaml"), []byte(body), 0o644)).To(Succeed())
}

func toYAML(cfg *config.ModelConfig) string {
	b, err := yaml.Marshal(cfg)
	Expect(err).NotTo(HaveOccurred())
	return string(b)
}

func openAIChat(content string) *schema.OpenAIRequest {
	req := &schema.OpenAIRequest{
		Messages: []schema.Message{
			{Role: "user", Content: content},
		},
	}
	req.Model = "smart-router"
	return req
}

func runRouter(loader *config.ModelConfigLoader, appConfig *config.ApplicationConfig, store router.DecisionStore, routerCfg *config.ModelConfig, parsed any, scorerFactory ScorerFactory) (*httptest.ResponseRecorder, error) {
	mw := RouteModel(loader, appConfig, store, nil, OpenAIProbe, router.SourceChat, ClassifierDeps{Scorer: scorerFactory})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("{}"))
	rec := httptest.NewRecorder()
	c := echo.New().NewContext(req, rec)
	c.Set(CONTEXT_LOCALS_KEY_MODEL_CONFIG, routerCfg)
	c.Set(CONTEXT_LOCALS_KEY_LOCALAI_REQUEST, parsed)
	handler := mw(func(c echo.Context) error {
		// Final hand-off — echo back which model the middleware
		// resolved so the spec can assert routing without exercising
		// the full chat pipeline.
		served, _ := c.Get(ContextKeyServedModel).(string)
		return c.String(http.StatusOK, "served:"+served)
	})
	err := handler(c)
	return rec, err
}

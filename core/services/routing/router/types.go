// Package router holds the routing module's classifier interface and
// the Score implementation.
//
// The dispatch architecture is: a "router model" in ModelConfig (one
// with a Router block) gets matched at request time. The classifier
// inspects the prompt and returns the set of policy labels it considers
// active; the surrounding middleware picks the first candidate whose
// labels are a superset of the active set, rewrites input.Model to that
// candidate, and falls back through the existing model resolution path.
// This keeps ACL checks, disabled-state, and per-model PII consistent —
// the router does *model* selection, nothing else.
//
// The package deliberately has no dependency on core/http or
// core/services — those wire the classifier in and feed it the request
// shape they own. Keeps the classifier easy to unit-test against
// synthetic Probe inputs and reusable from non-HTTP entry points
// (e.g., a future MCP routing tool).
package router

import (
	"context"
	"time"
)

// Probe is the classifier's input — the parsed prompt content the
// classifier needs to make a decision. Populated by the caller (the
// middleware does the schema-shape extraction); the classifier never
// inspects the original request struct.
type Probe struct {
	// Prompt is the merged user-visible text. For chat completions it
	// is the concatenation of message contents (separated by newlines);
	// for plain completions it is the raw prompt.
	Prompt string
}

// Decision is the classifier's output. Labels carries the SET of
// policy labels the classifier considers active for this probe. The
// surrounding middleware picks the first candidate whose Labels
// superset the active label set; that lets one prompt activate multiple
// policies and route to a model capable of all of them. Score is the
// softmax probability of the top label — kept for the decision log so
// admins can spot uncertain calls.
type Decision struct {
	Labels  []string      `json:"labels"`
	Score   float64       `json:"score"`
	Latency time.Duration `json:"latency"`

	// Cached is true when the decision came from the L2 embedding
	// cache rather than a fresh classifier run. CacheSimilarity carries
	// the cosine similarity of the cache hit (0 when not cached).
	Cached          bool    `json:"cached,omitempty"`
	CacheSimilarity float64 `json:"cache_similarity,omitempty"`
}

// Classifier is the entry point the middleware calls. The
// implementation honours ctx cancellation so long-running classifiers
// abort when the request context dies.
type Classifier interface {
	Classify(ctx context.Context, p Probe) (Decision, error)
	// Name is a stable identifier that ends up in RouterDecision rows
	// — admins read this to know which classifier produced a given
	// decision.
	Name() string
}

// Classifier names. Single source of truth for the YAML
// classifier: field, the buildClassifier dispatch in the
// middleware, and the strings each Classifier returns from Name().
const (
	// ClassifierScore picks labels by asking a small classifier
	// model (Arch-Router-style) to score each policy label as a
	// continuation of the routing prompt. See router/score.go for
	// the full rationale.
	ClassifierScore = "score"

	// ClassifierColbert picks labels by reranking each policy's
	// description against the prompt via LocalAI's rerankers
	// backend. Robust when policy labels are abstract relative to
	// user prompts — the description is the natural English the
	// reranker was trained on. The classifier_model points to a
	// reranker model (cross-encoder or bge-m3-colbert); the
	// `type:` field on that model's YAML controls which Reranker
	// library mode loads. See router/rerank.go.
	ClassifierColbert = "colbert"
)

// LabelFallback is the synthetic label written to the decision
// store when the middleware uses cfg.Router.Fallback rather than a
// classifier-picked candidate.
const LabelFallback = "fallback"

// errDecision packages an error with a populated Latency so each
// classifier's Classify can return early without restating the
// `Decision{Latency: time.Since(start)}, err` pattern.
func errDecision(start time.Time, err error) (Decision, error) {
	return Decision{Latency: time.Since(start)}, err
}

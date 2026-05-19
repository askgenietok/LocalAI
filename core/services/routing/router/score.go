package router

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/mudler/LocalAI/core/backend"
)

// ScorePolicy mirrors config.RouterPolicy at the classifier boundary —
// a label string plus its natural-language description for the
// routing system prompt.
type ScorePolicy struct {
	Label       string
	Description string
}

// defaultActivationThreshold is the softmax-probability floor a policy
// must clear to be considered "active." Picked low enough that two
// reasonably-confident labels (each ~0.4) both activate, high enough
// that a flat distribution doesn't activate everything.
const defaultActivationThreshold = 0.15

// ScoreClassifier scores every policy label as a continuation of the
// routing prompt, converts log-probabilities into a softmax
// distribution, and returns the set of labels whose probability
// passes the activation threshold.
//
// This is the off-the-shelf-Arch-Router approach extended for multi-
// label. The classifier model is trained to emit a single policy
// label, but its output distribution still spreads probability mass
// across competing labels when more than one applies. Reading the
// distribution rather than the argmax lets us route conjunctive
// intents ("debug this code AND explain the math") to a candidate
// that can serve both.
type ScoreClassifier struct {
	scorer              backend.Scorer
	activationThreshold float64

	// systemPrompt is built once at construction. The same prompt is
	// reused on every classification — only the user-turn body changes.
	systemPrompt string

	// labelOrder mirrors the configured policy ordering — the scorer
	// receives candidates in this order and the softmax distribution
	// indexes back into it.
	labelOrder []string

	cache *labelSetCache
}

// NewScoreClassifier panics on caller errors at construction (empty
// policies, missing description, nil scorer) — same rationale as the
// other classifiers. cacheCap=0 disables the cache.
// activationThreshold=0 picks the package default (0.15).
func NewScoreClassifier(policies []ScorePolicy, scorer backend.Scorer, cacheCap int, activationThreshold float64) *ScoreClassifier {
	if len(policies) == 0 {
		panic("router/score: at least one policy is required")
	}
	if scorer == nil {
		panic("router/score: scorer is required (configure router.classifier_model)")
	}
	for _, p := range policies {
		if p.Label == "" {
			panic("router/score: policy has empty label")
		}
		if p.Description == "" {
			panic(fmt.Sprintf("router/score: policy %q has no description", p.Label))
		}
	}
	labels := make([]string, 0, len(policies))
	for _, p := range policies {
		labels = append(labels, p.Label)
	}
	if activationThreshold <= 0 {
		activationThreshold = defaultActivationThreshold
	}
	return &ScoreClassifier{
		scorer:              scorer,
		activationThreshold: activationThreshold,
		systemPrompt:        buildScoreSystemPrompt(policies),
		labelOrder:          labels,
		cache:               newLabelSetCache(cacheCap),
	}
}

func (c *ScoreClassifier) Name() string { return ClassifierScore }

func (c *ScoreClassifier) Classify(ctx context.Context, p Probe) (Decision, error) {
	start := time.Now()
	key := cacheKey(p.Prompt)
	if hit, ok := c.cache.get(key); ok {
		return Decision{Labels: hit, Score: 1.0, Latency: time.Since(start)}, nil
	}
	prompt := buildScorePrompt(c.systemPrompt, p.Prompt)
	results, err := c.scorer.Score(ctx, prompt, c.labelOrder)
	if err != nil {
		return errDecision(start, fmt.Errorf("score classify: %w", err))
	}
	if len(results) != len(c.labelOrder) {
		return errDecision(start, fmt.Errorf("score classify: scorer returned %d results for %d policies", len(results), len(c.labelOrder)))
	}

	// Length-normalise log-probabilities (so candidates of unequal
	// token length stay comparable) then softmax to probabilities
	// suitable for thresholding.
	logProbs := make([]float64, len(results))
	for i, r := range results {
		switch {
		case r.NumTokens == 0:
			logProbs[i] = math.Inf(-1)
		case r.LengthNormalizedLogProb != 0:
			logProbs[i] = r.LengthNormalizedLogProb
		default:
			logProbs[i] = r.LogProb / float64(r.NumTokens)
		}
	}
	probs := softmax(logProbs)

	active, bestIdx := selectActive(probs, c.labelOrder, c.activationThreshold)
	c.cache.put(key, active)
	return Decision{
		Labels:  active,
		Score:   probs[bestIdx],
		Latency: time.Since(start),
	}, nil
}

// softmax converts an array of log-probabilities into a probability
// distribution. -inf inputs are handled (their exp contributes 0).
// Uses the standard max-subtraction trick for numerical stability.
func softmax(logProbs []float64) []float64 {
	if len(logProbs) == 0 {
		return nil
	}
	maxLP := math.Inf(-1)
	for _, lp := range logProbs {
		if lp > maxLP {
			maxLP = lp
		}
	}
	if math.IsInf(maxLP, -1) {
		// All -inf: return a uniform distribution as a sensible
		// degenerate result.
		out := make([]float64, len(logProbs))
		for i := range out {
			out[i] = 1.0 / float64(len(logProbs))
		}
		return out
	}
	out := make([]float64, len(logProbs))
	sum := 0.0
	for i, lp := range logProbs {
		out[i] = math.Exp(lp - maxLP)
		sum += out[i]
	}
	if sum == 0 {
		// Shouldn't happen given the maxLP check above, but guard
		// against pathological inputs.
		for i := range out {
			out[i] = 1.0 / float64(len(out))
		}
		return out
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func (c *ScoreClassifier) CacheLen() int { return c.cache.len() }

func buildScoreSystemPrompt(policies []ScorePolicy) string {
	var b strings.Builder
	b.WriteString("You are a routing classifier. Pick the policy whose description best matches the user's request.\n\n")
	b.WriteString("Available policies:\n")
	for _, p := range policies {
		b.WriteString("- ")
		b.WriteString(p.Label)
		b.WriteString(": ")
		b.WriteString(p.Description)
		b.WriteString("\n")
	}
	return b.String()
}

// buildScorePrompt assembles the Qwen/ChatML-style prompt the
// Arch-Router model was trained on. The candidate label is scored as
// the assistant's first token(s) of response — so we end the prompt
// right at the assistant-turn marker, no trailing newline.
//
// Hard-coded to ChatML for now: Arch-Router is Qwen-2.5-1.5B-Instruct
// based and the published GGUF carries this template natively. When
// we add a non-ChatML scoring model we'll thread the template through
// from ModelConfig.
func buildScorePrompt(system, user string) string {
	var b strings.Builder
	b.WriteString("<|im_start|>system\n")
	b.WriteString(system)
	b.WriteString("<|im_end|>\n<|im_start|>user\n")
	b.WriteString(user)
	b.WriteString("<|im_end|>\n<|im_start|>assistant\n")
	return b.String()
}

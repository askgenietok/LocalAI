package router

import (
	"context"
	"errors"
	"sort"
	"strings"

	"github.com/mudler/LocalAI/core/backend"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

type stubScorer struct {
	results []backend.CandidateScore
	err     error
	calls   int
	lastP   string
	lastC   []string
}

func (s *stubScorer) Score(_ context.Context, prompt string, candidates []string) ([]backend.CandidateScore, error) {
	s.calls++
	s.lastP = prompt
	s.lastC = append(s.lastC[:0], candidates...)
	if s.err != nil {
		return nil, s.err
	}
	return s.results, nil
}

func testPolicies() []ScorePolicy {
	return []ScorePolicy{
		{Label: "code-generation", Description: "writing, debugging, or explaining code"},
		{Label: "casual-chat", Description: "small talk and general conversation"},
		{Label: "math-reasoning", Description: "arithmetic, equations, word problems"},
	}
}

func sortedLabels(d Decision) []string {
	out := append([]string(nil), d.Labels...)
	sort.Strings(out)
	return out
}

func equalLabels(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

var _ = Describe("ScoreClassifier", func() {
	It("returns a single dominant label", func() {
		// A confident single-label classification: code-generation
		// dominates softmax, the others fall well below the activation
		// threshold (default 0.15).
		s := &stubScorer{results: []backend.CandidateScore{
			{LogProb: -0.05, LengthNormalizedLogProb: -0.025, NumTokens: 2}, // code
			{LogProb: -8.0, LengthNormalizedLogProb: -2.667, NumTokens: 3},  // chat
			{LogProb: -10.0, LengthNormalizedLogProb: -2.5, NumTokens: 4},   // math
		}}
		c := NewScoreClassifier(testPolicies(), s, 0, 0)
		d, err := c.Classify(context.Background(), Probe{Prompt: "fix this null pointer"})
		Expect(err).NotTo(HaveOccurred(), "Classify")
		Expect(equalLabels(d.Labels, []string{"code-generation"})).To(BeTrue(), "Labels = %v, want [code-generation]", d.Labels)
		// Score is the top softmax probability. Two ~-2.5 distractors
		// vs a ~0 winner gives ~0.86 for the winner — high enough to
		// signal confidence in the decision log.
		Expect(d.Score).To(BeNumerically(">=", 0.8), "want >= 0.8 for dominant single label")
	})

	It("activates multiple labels", func() {
		// Two-way tie: code and math each take ~0.5 of the probability
		// mass, chat is far behind. Both labels must activate so the
		// router can pick a candidate covering both capabilities.
		s := &stubScorer{results: []backend.CandidateScore{
			{LogProb: -2.0, LengthNormalizedLogProb: -1.0, NumTokens: 2}, // code  ~0.49
			{LogProb: -9.0, LengthNormalizedLogProb: -3.0, NumTokens: 3}, // chat  ~0.01
			{LogProb: -4.0, LengthNormalizedLogProb: -1.0, NumTokens: 4}, // math  ~0.49
		}}
		c := NewScoreClassifier(testPolicies(), s, 0, 0)
		d, err := c.Classify(context.Background(), Probe{Prompt: "write code that solves this word problem"})
		Expect(err).NotTo(HaveOccurred(), "Classify")
		got := sortedLabels(d)
		want := []string{"code-generation", "math-reasoning"}
		Expect(equalLabels(got, want)).To(BeTrue(), "Labels = %v, want %v", got, want)
	})

	It("falls back to argmax on flat distribution", func() {
		// All three labels score roughly equally. Strict
		// activation-threshold filtering could return zero labels, which
		// would leave the router with nothing to match. The classifier
		// falls back to argmax in this case so callers always have at
		// least one label to route on.
		s := &stubScorer{results: []backend.CandidateScore{
			{LogProb: -2.0, LengthNormalizedLogProb: -1.0, NumTokens: 2}, // ~0.33
			{LogProb: -3.0, LengthNormalizedLogProb: -1.0, NumTokens: 3}, // ~0.33
			{LogProb: -4.0, LengthNormalizedLogProb: -1.0, NumTokens: 4}, // ~0.33
		}}
		// Threshold above max softmax probability (0.5) forces the
		// fallback path.
		c := NewScoreClassifier(testPolicies(), s, 0, 0.5)
		d, err := c.Classify(context.Background(), Probe{Prompt: "x"})
		Expect(err).NotTo(HaveOccurred(), "Classify")
		Expect(d.Labels).To(HaveLen(1), "want fallback to argmax (single label)")
	})

	It("falls back to joint log-prob when length normalisation missing", func() {
		// Backend that doesn't honour length_normalize — only LogProb is
		// populated. The classifier derives the per-token score itself
		// so candidates of different token lengths stay comparable. If
		// it didn't, the joint log-probs (-8, -5, -6) would pick chat —
		// purely because it has fewer tokens. With length-norm chat is
		// behind on per-token quality.
		s := &stubScorer{results: []backend.CandidateScore{
			{LogProb: -8.0, NumTokens: 4},  // -2.0 per token
			{LogProb: -15.0, NumTokens: 2}, // -7.5 per token — clearly out
			{LogProb: -6.0, NumTokens: 3},  // -2.0 per token
		}}
		c := NewScoreClassifier(testPolicies(), s, 0, 0)
		d, err := c.Classify(context.Background(), Probe{Prompt: "x"})
		Expect(err).NotTo(HaveOccurred(), "Classify")
		got := sortedLabels(d)
		want := []string{"code-generation", "math-reasoning"}
		Expect(equalLabels(got, want)).To(BeTrue(), "Labels = %v, want %v", got, want)
	})

	It("builds ChatML prompt with system and user", func() {
		s := &stubScorer{results: []backend.CandidateScore{
			{LogProb: -1, LengthNormalizedLogProb: -0.5, NumTokens: 2},
			{LogProb: -2, LengthNormalizedLogProb: -0.67, NumTokens: 3},
			{LogProb: -3, LengthNormalizedLogProb: -0.75, NumTokens: 4},
		}}
		c := NewScoreClassifier(testPolicies(), s, 0, 0)
		_, err := c.Classify(context.Background(), Probe{Prompt: "hello world"})
		Expect(err).NotTo(HaveOccurred(), "Classify")
		Expect(s.lastP).To(ContainSubstring("<|im_start|>system"))
		Expect(s.lastP).To(ContainSubstring("code-generation: writing, debugging"))
		Expect(s.lastP).To(ContainSubstring("<|im_start|>user\nhello world<|im_end|>"))
		Expect(strings.HasSuffix(s.lastP, "<|im_start|>assistant\n")).To(BeTrue(), "prompt does not end at assistant marker: %q", s.lastP)
		Expect(s.lastC).To(HaveLen(3))
		Expect(s.lastC[0]).To(Equal("code-generation"))
		Expect(s.lastC[1]).To(Equal("casual-chat"))
		Expect(s.lastC[2]).To(Equal("math-reasoning"))
	})

	It("caches by normalised prompt", func() {
		s := &stubScorer{results: []backend.CandidateScore{
			{LogProb: -0.1, LengthNormalizedLogProb: -0.05, NumTokens: 2},
			{LogProb: -5, LengthNormalizedLogProb: -1.67, NumTokens: 3},
			{LogProb: -6, LengthNormalizedLogProb: -1.5, NumTokens: 4},
		}}
		c := NewScoreClassifier(testPolicies(), s, 64, 0)
		_, err := c.Classify(context.Background(), Probe{Prompt: "Fix Bug"})
		Expect(err).NotTo(HaveOccurred(), "classify 1")
		_, err = c.Classify(context.Background(), Probe{Prompt: " fix bug "})
		Expect(err).NotTo(HaveOccurred(), "classify 2")
		Expect(s.calls).To(Equal(1), "second classify should hit cache")
		Expect(c.CacheLen()).To(Equal(1))
	})

	It("cache disabled when cap zero", func() {
		s := &stubScorer{results: []backend.CandidateScore{
			{LogProb: -1, LengthNormalizedLogProb: -0.5, NumTokens: 2},
			{LogProb: -5, LengthNormalizedLogProb: -1.67, NumTokens: 3},
			{LogProb: -6, LengthNormalizedLogProb: -1.5, NumTokens: 4},
		}}
		c := NewScoreClassifier(testPolicies(), s, 0, 0)
		for i := 0; i < 3; i++ {
			_, err := c.Classify(context.Background(), Probe{Prompt: "same"})
			Expect(err).NotTo(HaveOccurred(), "classify")
		}
		Expect(s.calls).To(Equal(3), "cache disabled")
	})

	It("propagates scorer error", func() {
		scorerErr := errors.New("boom")
		c := NewScoreClassifier(testPolicies(), &stubScorer{err: scorerErr}, 0, 0)
		_, err := c.Classify(context.Background(), Probe{Prompt: "x"})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("boom"), "expected scorer error to propagate")
	})

	It("returns result-count mismatch as error", func() {
		s := &stubScorer{results: []backend.CandidateScore{
			{LogProb: -1, LengthNormalizedLogProb: -0.5, NumTokens: 2},
		}}
		c := NewScoreClassifier(testPolicies(), s, 0, 0)
		_, err := c.Classify(context.Background(), Probe{Prompt: "x"})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("returned 1 results for 3 policies"))
	})

	It("zero-token candidate scores -inf", func() {
		// A NumTokens=0 candidate must contribute zero softmax mass and
		// never win, even if its raw log-prob looks favourable.
		s := &stubScorer{results: []backend.CandidateScore{
			{LogProb: 100, LengthNormalizedLogProb: 100, NumTokens: 0}, // degenerate
			{LogProb: -2, LengthNormalizedLogProb: -1.0, NumTokens: 2},
			{LogProb: -3, LengthNormalizedLogProb: -0.75, NumTokens: 4},
		}}
		c := NewScoreClassifier(testPolicies(), s, 0, 0)
		d, err := c.Classify(context.Background(), Probe{Prompt: "x"})
		Expect(err).NotTo(HaveOccurred(), "Classify")
		for _, l := range d.Labels {
			Expect(l).NotTo(Equal("code-generation"), "NumTokens=0 label must not be active")
		}
	})

	It("panics on empty policies", func() {
		Expect(func() { NewScoreClassifier(nil, &stubScorer{}, 0, 0) }).To(Panic())
	})

	It("panics on nil scorer", func() {
		Expect(func() { NewScoreClassifier(testPolicies(), nil, 0, 0) }).To(Panic())
	})

	It("panics on missing description", func() {
		Expect(func() { NewScoreClassifier([]ScorePolicy{{Label: "x"}}, &stubScorer{}, 0, 0) }).To(Panic())
	})

	It("Name returns the classifier identifier", func() {
		c := NewScoreClassifier(testPolicies(), &stubScorer{}, 0, 0)
		Expect(c.Name()).To(Equal(ClassifierScore))
	})
})

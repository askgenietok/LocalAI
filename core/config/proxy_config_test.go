package config

import (
	"os"
	"path/filepath"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("ProxyConfig.ResolveAPIKey", func() {
	It("returns empty for no-auth upstream", func() {
		key, err := ProxyConfig{}.ResolveAPIKey()
		Expect(err).NotTo(HaveOccurred())
		Expect(key).To(BeEmpty())
	})

	It("reads from environment when api_key_env is set", func() {
		GinkgoT().Setenv("PROXY_TEST_RESOLVE_KEY", "sk-from-env")
		key, err := ProxyConfig{APIKeyEnv: "PROXY_TEST_RESOLVE_KEY"}.ResolveAPIKey()
		Expect(err).NotTo(HaveOccurred())
		Expect(key).To(Equal("sk-from-env"))
	})

	It("returns an error when api_key_env is set but unset in the environment", func() {
		// Make sure the var really isn't set.
		os.Unsetenv("PROXY_TEST_RESOLVE_UNSET")
		_, err := ProxyConfig{APIKeyEnv: "PROXY_TEST_RESOLVE_UNSET"}.ResolveAPIKey()
		Expect(err).To(MatchError(ContainSubstring("unset")))
	})

	It("reads from file when api_key_file is set", func() {
		dir := GinkgoT().TempDir()
		path := filepath.Join(dir, "secret.txt")
		// Trailing newline must be trimmed — secret stores often add one.
		Expect(os.WriteFile(path, []byte("sk-from-file\n"), 0600)).To(Succeed())

		key, err := ProxyConfig{APIKeyFile: path}.ResolveAPIKey()
		Expect(err).NotTo(HaveOccurred())
		Expect(key).To(Equal("sk-from-file"))
	})

	It("returns an error when api_key_file path does not exist", func() {
		_, err := ProxyConfig{APIKeyFile: "/nonexistent/path/xyz"}.ResolveAPIKey()
		Expect(err).To(MatchError(ContainSubstring("read api_key_file")))
	})

	It("prefers api_key_env over api_key_file when both are set", func() {
		// Validate forbids this combination, but ResolveAPIKey is also
		// called in pathways where Validate hasn't run (defensive read
		// from a partially-constructed config). Document the actual
		// precedence so a future change can't silently flip it.
		GinkgoT().Setenv("PROXY_TEST_RESOLVE_BOTH", "env-wins")
		dir := GinkgoT().TempDir()
		path := filepath.Join(dir, "secret.txt")
		Expect(os.WriteFile(path, []byte("file-loses"), 0600)).To(Succeed())

		key, err := ProxyConfig{
			APIKeyEnv:  "PROXY_TEST_RESOLVE_BOTH",
			APIKeyFile: path,
		}.ResolveAPIKey()
		Expect(err).NotTo(HaveOccurred())
		Expect(key).To(Equal("env-wins"))
	})
})

var _ = Describe("ModelConfig.SetDefaults proxy normalisation", func() {
	It("normalises empty Mode to passthrough", func() {
		cfg := &ModelConfig{Backend: "cloud-proxy"}
		cfg.SetDefaults()
		Expect(cfg.Proxy.Mode).To(Equal(ProxyModePassthrough))
	})

	It("leaves a non-empty Mode untouched", func() {
		cfg := &ModelConfig{
			Backend: "cloud-proxy",
			Proxy:   ProxyConfig{Mode: ProxyModeTranslate, Provider: ProxyProviderOpenAI},
		}
		cfg.SetDefaults()
		Expect(cfg.Proxy.Mode).To(Equal(ProxyModeTranslate))
	})

	It("normalises Mode even on non-cloud-proxy backends", func() {
		// Defensive: the normalisation runs unconditionally so downstream
		// code can switch on cfg.Proxy.Mode without nil-checking the
		// backend type. A future cloud-proxy-only invariant would change
		// this — keep it tested either way.
		cfg := &ModelConfig{Backend: "llama-cpp"}
		cfg.SetDefaults()
		Expect(cfg.Proxy.Mode).To(Equal(ProxyModePassthrough))
	})
})

var _ = Describe("ModelConfig.Validate proxy rules", func() {
	baseCloudProxy := func() *ModelConfig {
		return &ModelConfig{
			Name:    "cp-test",
			Backend: "cloud-proxy",
			Proxy: ProxyConfig{
				UpstreamURL: "https://api.openai.com/v1/chat/completions",
			},
		}
	}

	It("accepts empty Mode (defaulted later)", func() {
		cfg := baseCloudProxy()
		cfg.Proxy.Mode = ""
		ok, err := cfg.Validate()
		Expect(err).NotTo(HaveOccurred())
		Expect(ok).To(BeTrue())
	})

	It("accepts passthrough mode", func() {
		cfg := baseCloudProxy()
		cfg.Proxy.Mode = ProxyModePassthrough
		ok, err := cfg.Validate()
		Expect(err).NotTo(HaveOccurred())
		Expect(ok).To(BeTrue())
	})

	It("accepts translate mode with provider", func() {
		cfg := baseCloudProxy()
		cfg.Proxy.Mode = ProxyModeTranslate
		cfg.Proxy.Provider = ProxyProviderAnthropic
		ok, err := cfg.Validate()
		Expect(err).NotTo(HaveOccurred())
		Expect(ok).To(BeTrue())
	})

	It("rejects unknown mode", func() {
		cfg := baseCloudProxy()
		cfg.Proxy.Mode = "rewrite"
		ok, err := cfg.Validate()
		Expect(ok).To(BeFalse())
		Expect(err).To(MatchError(ContainSubstring("unknown mode")))
	})

	It("rejects translate mode without provider", func() {
		cfg := baseCloudProxy()
		cfg.Proxy.Mode = ProxyModeTranslate
		cfg.Proxy.Provider = ""
		ok, err := cfg.Validate()
		Expect(ok).To(BeFalse())
		Expect(err).To(MatchError(ContainSubstring("translate mode requires provider")))
	})

	It("rejects both api_key_env and api_key_file set", func() {
		cfg := baseCloudProxy()
		cfg.Proxy.APIKeyEnv = "X"
		cfg.Proxy.APIKeyFile = "/tmp/y"
		ok, err := cfg.Validate()
		Expect(ok).To(BeFalse())
		Expect(err).To(MatchError(ContainSubstring("mutually exclusive")))
	})

	It("accepts no api_key_* set (no-auth upstream)", func() {
		cfg := baseCloudProxy()
		ok, err := cfg.Validate()
		Expect(err).NotTo(HaveOccurred())
		Expect(ok).To(BeTrue())
	})
})

var _ = Describe("ModelConfig.IsCloudProxyBackendPassthrough", func() {
	It("returns true for cloud-proxy with empty mode", func() {
		cfg := &ModelConfig{Backend: "cloud-proxy"}
		Expect(cfg.IsCloudProxyBackendPassthrough()).To(BeTrue())
	})

	It("returns true for cloud-proxy with explicit passthrough", func() {
		cfg := &ModelConfig{
			Backend: "cloud-proxy",
			Proxy:   ProxyConfig{Mode: ProxyModePassthrough},
		}
		Expect(cfg.IsCloudProxyBackendPassthrough()).To(BeTrue())
	})

	It("returns false for cloud-proxy in translate mode", func() {
		cfg := &ModelConfig{
			Backend: "cloud-proxy",
			Proxy:   ProxyConfig{Mode: ProxyModeTranslate, Provider: ProxyProviderOpenAI},
		}
		Expect(cfg.IsCloudProxyBackendPassthrough()).To(BeFalse())
	})

	It("returns false for non-cloud-proxy backends", func() {
		Expect((&ModelConfig{Backend: "llama-cpp"}).IsCloudProxyBackendPassthrough()).To(BeFalse())
	})
})

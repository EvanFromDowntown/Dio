package cli

import (
	"fmt"
	"strings"

	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/llm/anthropic"
	"github.com/EvanFromDowntown/Dio/llm/chatcompletions"
	"github.com/EvanFromDowntown/Dio/llm/openai"
)

// NewClient builds the provider client. The API key is read from the
// environment variable named by opts.APIKeyEnv.
func NewClient(opts *Options, getenv func(string) string) (llm.Client, error) {
	key := getenv(opts.APIKeyEnv)
	if key == "" {
		return nil, fmt.Errorf("环境变量 %s 未设置：请先 export %s=<API key>，或用 -api-key-env 指定其他变量", opts.APIKeyEnv, opts.APIKeyEnv)
	}
	return NewProviderClient(opts.Provider, opts.BaseURL, key)
}

// NewProviderClient builds a client for one of Providers.
func NewProviderClient(provider, baseURL, apiKey string) (llm.Client, error) {
	base := strings.TrimRight(baseURL, "/")
	switch provider {
	case "chatcompletions":
		o := []chatcompletions.Option{chatcompletions.WithAPIKey(apiKey)}
		if base != "" {
			o = append(o, chatcompletions.WithBaseURL(base))
		}
		return chatcompletions.New(o...), nil
	case "openai":
		o := []openai.Option{openai.WithAPIKey(apiKey)}
		if base != "" {
			o = append(o, openai.WithBaseURL(base))
		}
		return openai.New(o...), nil
	case "anthropic":
		o := []anthropic.Option{anthropic.WithAPIKey(apiKey)}
		if base != "" {
			// The adapter appends /v1/messages, so accept the same ".../v1"
			// base URL the OpenAI-style providers use.
			o = append(o, anthropic.WithBaseURL(strings.TrimSuffix(base, "/v1")))
		}
		return anthropic.New(o...), nil
	}
	return nil, fmt.Errorf("未知的 provider %q", provider)
}

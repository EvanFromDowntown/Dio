package cli

import (
	"fmt"
	"strings"

	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/llm/anthropic/messages"
	"github.com/EvanFromDowntown/Dio/llm/openai/chatcompletions"
	"github.com/EvanFromDowntown/Dio/llm/openai/responses"
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

// NewProviderClient builds a client for a provider ID (legacy names are
// accepted).
func NewProviderClient(provider, baseURL, apiKey string) (llm.Client, error) {
	base := strings.TrimRight(baseURL, "/")
	switch llm.NormalizeProvider(provider) {
	case chatcompletions.ProviderName:
		o := []chatcompletions.Option{chatcompletions.WithAPIKey(apiKey)}
		if base != "" {
			o = append(o, chatcompletions.WithBaseURL(base))
		}
		return chatcompletions.New(o...), nil
	case responses.ProviderName:
		o := []responses.Option{responses.WithAPIKey(apiKey)}
		if base != "" {
			o = append(o, responses.WithBaseURL(base))
		}
		return responses.New(o...), nil
	case messages.ProviderName:
		o := []messages.Option{messages.WithAPIKey(apiKey)}
		if base != "" {
			// The adapter appends /v1/messages, so accept the same ".../v1"
			// base URL the OpenAI-style providers use.
			o = append(o, messages.WithBaseURL(strings.TrimSuffix(base, "/v1")))
		}
		return messages.New(o...), nil
	}
	return nil, fmt.Errorf("未知的 provider %q，可选：%s", provider, strings.Join(llm.Providers, "、"))
}

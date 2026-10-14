package llm

// Provider IDs identify a wire protocol. Adapters use them for
// Reasoning.Provider and APIError.Provider, and sessions store them.
const (
	ProviderChatCompletions = "openai/chat-completions"
	ProviderResponses       = "openai/responses"
	ProviderMessages        = "anthropic/messages"
)

// Providers lists the provider IDs in display order.
var Providers = []string{ProviderChatCompletions, ProviderResponses, ProviderMessages}

// legacyProviders maps IDs written by earlier versions to current ones.
var legacyProviders = map[string]string{
	"chatcompletions": ProviderChatCompletions,
	"openai":          ProviderResponses,
	"anthropic":       ProviderMessages,
}

// NormalizeProvider maps legacy provider names to current IDs and returns
// any other value unchanged.
func NormalizeProvider(name string) string {
	if id, ok := legacyProviders[name]; ok {
		return id
	}
	return name
}

// IsProvider reports whether name, after normalization, is a known ID.
func IsProvider(name string) bool {
	switch NormalizeProvider(name) {
	case ProviderChatCompletions, ProviderResponses, ProviderMessages:
		return true
	}
	return false
}

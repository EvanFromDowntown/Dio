package llm

import "testing"

func TestNormalizeProvider(t *testing.T) {
	tests := map[string]string{
		"chatcompletions":         ProviderChatCompletions,
		"openai":                  ProviderResponses,
		"anthropic":               ProviderMessages,
		"openai/chat-completions": "openai/chat-completions",
		"openai/responses":        "openai/responses",
		"anthropic/messages":      "anthropic/messages",
		"fake":                    "fake",
		"":                        "",
	}
	for in, want := range tests {
		if got := NormalizeProvider(in); got != want {
			t.Errorf("NormalizeProvider(%q) = %q, want %q", in, got, want)
		}
	}
	for _, name := range []string{"openai/chat-completions", "openai/responses", "anthropic/messages", "chatcompletions", "openai", "anthropic"} {
		if !IsProvider(name) {
			t.Errorf("IsProvider(%q) = false", name)
		}
	}
	for _, name := range []string{"", "gemini", "openai/chat_completions", "OpenAI/Responses"} {
		if IsProvider(name) {
			t.Errorf("IsProvider(%q) = true", name)
		}
	}
	if len(Providers) != 3 || Providers[0] != "openai/chat-completions" || Providers[1] != "openai/responses" || Providers[2] != "anthropic/messages" {
		t.Errorf("Providers = %v", Providers)
	}
}

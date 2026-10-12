package llm

import (
	"errors"
	"fmt"
)

// Request is a provider-agnostic generation request.
type Request struct {
	Model    string    `json:"model"`
	System   string    `json:"system,omitempty"`
	Messages []Message `json:"messages"`

	Tools      []Tool      `json:"tools,omitempty"`
	ToolChoice *ToolChoice `json:"tool_choice,omitempty"`

	// MaxTokens caps output tokens. Zero means the adapter default.
	MaxTokens     int      `json:"max_tokens,omitempty"`
	Temperature   *float64 `json:"temperature,omitempty"`
	TopP          *float64 `json:"top_p,omitempty"`
	StopSequences []string `json:"stop_sequences,omitempty"`

	Reasoning *ReasoningConfig `json:"reasoning,omitempty"`

	// Extra is merged into the top level of the provider request body after
	// standard fields are encoded, overriding them on key collisions. Use it
	// for provider- or gateway-specific parameters.
	Extra map[string]any `json:"extra,omitempty"`
}

// ReasoningConfig enables extended thinking / reasoning.
type ReasoningConfig struct {
	// Effort is "low", "medium" or "high" (OpenAI reasoning.effort).
	Effort string `json:"effort,omitempty"`
	// BudgetTokens is the Anthropic thinking budget. When zero, Anthropic
	// derives a budget from Effort.
	BudgetTokens int `json:"budget_tokens,omitempty"`
	// Summary requests an OpenAI reasoning summary ("auto", "concise",
	// "detailed").
	Summary string `json:"summary,omitempty"`
}

// Validate checks provider-independent invariants.
func (r *Request) Validate() error {
	if r == nil {
		return errors.New("llm: nil request")
	}
	if r.Model == "" {
		return errors.New("llm: request model is required")
	}
	if len(r.Messages) == 0 {
		return errors.New("llm: request needs at least one message")
	}
	for i, m := range r.Messages {
		if m.Role != RoleUser && m.Role != RoleAssistant {
			return fmt.Errorf("llm: message %d has invalid role %q", i, m.Role)
		}
	}
	if tc := r.ToolChoice; tc != nil && tc.Mode == ToolChoiceTool && tc.Name == "" {
		return errors.New("llm: tool choice mode \"tool\" requires a name")
	}
	return nil
}

// StopReason is the normalized reason generation ended.
type StopReason string

const (
	StopEndTurn       StopReason = "end_turn"
	StopMaxTokens     StopReason = "max_tokens"
	StopSequence      StopReason = "stop_sequence"
	StopToolUse       StopReason = "tool_use"
	StopContentFilter StopReason = "content_filter"
	StopOther         StopReason = "other"
)

// Usage reports token consumption. InputTokens is the total prompt size,
// including cached tokens; the cache fields break that total down.
type Usage struct {
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	CacheReadTokens  int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
	// ReasoningTokens is the part of OutputTokens spent on reasoning, when
	// the provider reports it.
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// Response is a completed assistant turn.
type Response struct {
	ID         string         `json:"id"`
	Model      string         `json:"model"`
	Content    []ContentBlock `json:"content"`
	StopReason StopReason     `json:"stop_reason"`
	// ProviderStopReason is the raw provider value StopReason was derived from.
	ProviderStopReason string `json:"provider_stop_reason,omitempty"`
	Usage              Usage  `json:"usage"`
}

// Message returns the response as an assistant message, ready to append to
// the conversation history.
func (r *Response) Message() Message {
	return Message{Role: RoleAssistant, Content: r.Content}
}

func (r *Response) Text() string { return r.Message().Text() }

func (r *Response) ToolCalls() []ToolCall { return r.Message().ToolCalls() }

// Ptr returns a pointer to v, for optional request fields like Temperature.
func Ptr[T any](v T) *T { return &v }

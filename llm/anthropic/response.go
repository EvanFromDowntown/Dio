package anthropic

import (
	"encoding/json"

	"github.com/EvanFromDowntown/Dio/llm"
)

type apiResponse struct {
	ID         string     `json:"id"`
	Model      string     `json:"model"`
	Content    []apiBlock `json:"content"`
	StopReason string     `json:"stop_reason"`
	Usage      apiUsage   `json:"usage"`
}

type apiBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	Thinking  string          `json:"thinking"`
	Signature string          `json:"signature"`
	Data      string          `json:"data"`
}

type apiUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

func convertResponse(r *apiResponse) *llm.Response {
	out := &llm.Response{
		ID:      r.ID,
		Model:   r.Model,
		Content: make([]llm.ContentBlock, 0, len(r.Content)),
		Usage:   convertUsage(r.Usage),
	}
	for _, b := range r.Content {
		if block, ok := convertBlock(b); ok {
			out.Content = append(out.Content, block)
		}
	}
	out.StopReason, out.ProviderStopReason = convertStopReason(r.StopReason), r.StopReason
	return out
}

func convertBlock(b apiBlock) (llm.ContentBlock, bool) {
	switch b.Type {
	case "text":
		return llm.TextBlock(b.Text), true
	case "tool_use":
		input := b.Input
		if len(input) == 0 {
			input = json.RawMessage(`{}`)
		}
		return llm.ToolCallBlock(llm.ToolCall{ID: b.ID, Name: b.Name, Arguments: input}), true
	case "thinking":
		return llm.ContentBlock{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{
			Provider: ProviderName, Text: b.Thinking, Signature: b.Signature,
		}}, true
	case "redacted_thinking":
		return llm.ContentBlock{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{
			Provider: ProviderName, Encrypted: b.Data,
		}}, true
	}
	return llm.ContentBlock{}, false
}

// convertUsage normalizes InputTokens to include cached tokens; Anthropic
// reports uncached input separately from cache reads and writes.
func convertUsage(u apiUsage) llm.Usage {
	return llm.Usage{
		InputTokens:      u.InputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens,
		OutputTokens:     u.OutputTokens,
		CacheReadTokens:  u.CacheReadInputTokens,
		CacheWriteTokens: u.CacheCreationInputTokens,
	}
}

func convertStopReason(s string) llm.StopReason {
	switch s {
	case "end_turn":
		return llm.StopEndTurn
	case "max_tokens":
		return llm.StopMaxTokens
	case "stop_sequence":
		return llm.StopSequence
	case "tool_use":
		return llm.StopToolUse
	case "refusal":
		return llm.StopContentFilter
	}
	return llm.StopOther
}

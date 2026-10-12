package chatcompletions

import (
	"encoding/json"
	"errors"
	"strings"

	"github.com/EvanFromDowntown/Dio/llm"
)

type apiResponse struct {
	ID      string      `json:"id"`
	Model   string      `json:"model"`
	Choices []apiChoice `json:"choices"`
	Usage   *apiUsage   `json:"usage"`
	Error   *struct {
		Type    string          `json:"type"`
		Code    json.RawMessage `json:"code"`
		Message string          `json:"message"`
	} `json:"error"`
}

type apiChoice struct {
	Index        int        `json:"index"`
	Message      *apiOutput `json:"message"`
	Delta        *apiOutput `json:"delta"`
	FinishReason string     `json:"finish_reason"`
}

// apiOutput is both a full message and a streaming delta.
type apiOutput struct {
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content"`
	ReasoningContent string          `json:"reasoning_content"`
	Refusal          string          `json:"refusal"`
	ToolCalls        []apiToolCall   `json:"tool_calls"`
}

type apiUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
	// DeepSeek reports cache hits here instead of prompt_tokens_details.
	PromptCacheHitTokens int `json:"prompt_cache_hit_tokens"`
}

func convertResponse(r *apiResponse) (*llm.Response, error) {
	if r.Error != nil && len(r.Choices) == 0 {
		apiErr := &llm.APIError{Provider: ProviderName, Type: r.Error.Type, Message: r.Error.Message}
		var code string
		if json.Unmarshal(r.Error.Code, &code) != nil {
			code = strings.Trim(string(r.Error.Code), `"`)
		}
		if code != "null" {
			apiErr.Code = code
		}
		return nil, apiErr
	}
	if len(r.Choices) == 0 || r.Choices[0].Message == nil {
		return nil, errors.New("chatcompletions: response has no choices")
	}
	choice := r.Choices[0]
	m := choice.Message
	out := &llm.Response{ID: r.ID, Model: r.Model, Content: []llm.ContentBlock{}}
	if m.ReasoningContent != "" {
		out.Content = append(out.Content, reasoningBlock(m.ReasoningContent))
	}
	if text := contentText(m.Content); text != "" {
		out.Content = append(out.Content, llm.TextBlock(text))
	} else if m.Refusal != "" {
		out.Content = append(out.Content, llm.TextBlock(m.Refusal))
	}
	for _, tc := range m.ToolCalls {
		out.Content = append(out.Content, toolCallBlock(tc.ID, tc.Function.Name, tc.Function.Arguments))
	}
	out.StopReason, out.ProviderStopReason = convertFinishReason(choice.FinishReason), choice.FinishReason
	if r.Usage != nil {
		out.Usage = convertUsage(r.Usage)
	}
	return out, nil
}

// contentText accepts content as a string, null, or an array of text parts.
func contentText(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []contentPart
	if json.Unmarshal(raw, &parts) != nil {
		return ""
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

func reasoningBlock(text string) llm.ContentBlock {
	return llm.ContentBlock{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: ProviderName, Text: text}}
}

func toolCallBlock(id, name, args string) llm.ContentBlock {
	if args == "" {
		args = "{}"
	}
	return llm.ToolCallBlock(llm.ToolCall{ID: id, Name: name, Arguments: json.RawMessage(args)})
}

// convertFinishReason cannot distinguish a matched stop sequence from a
// natural end: both are reported as "stop".
func convertFinishReason(s string) llm.StopReason {
	switch s {
	case "stop":
		return llm.StopEndTurn
	case "length":
		return llm.StopMaxTokens
	case "tool_calls", "function_call":
		return llm.StopToolUse
	case "content_filter":
		return llm.StopContentFilter
	}
	return llm.StopOther
}

func convertUsage(u *apiUsage) llm.Usage {
	out := llm.Usage{
		InputTokens:     u.PromptTokens,
		OutputTokens:    u.CompletionTokens,
		CacheReadTokens: u.PromptCacheHitTokens,
	}
	if d := u.PromptTokensDetails; d != nil && d.CachedTokens > out.CacheReadTokens {
		out.CacheReadTokens = d.CachedTokens
	}
	if d := u.CompletionTokensDetails; d != nil {
		out.ReasoningTokens = d.ReasoningTokens
	}
	return out
}

package openai

import (
	"encoding/json"
	"strings"

	"github.com/EvanFromDowntown/Dio/llm"
)

type apiResponse struct {
	ID                string       `json:"id"`
	Model             string       `json:"model"`
	Status            string       `json:"status"`
	Output            []outputItem `json:"output"`
	Usage             *apiUsage    `json:"usage"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

type outputItem struct {
	Type             string          `json:"type"`
	ID               string          `json:"id"`
	Role             string          `json:"role"`
	Content          []outputContent `json:"content"`
	CallID           string          `json:"call_id"`
	Name             string          `json:"name"`
	Arguments        string          `json:"arguments"`
	Summary          []summaryPart   `json:"summary"`
	EncryptedContent string          `json:"encrypted_content"`
}

type outputContent struct {
	Type    string `json:"type"`
	Text    string `json:"text"`
	Refusal string `json:"refusal"`
}

type apiUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	InputTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	OutputTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}

func convertResponse(r *apiResponse) (*llm.Response, error) {
	if r.Status == "failed" {
		apiErr := &llm.APIError{Provider: ProviderName, Message: "response failed"}
		if r.Error != nil {
			apiErr.Code = r.Error.Code
			apiErr.Message = r.Error.Message
		}
		return nil, apiErr
	}
	out := &llm.Response{ID: r.ID, Model: r.Model, Content: []llm.ContentBlock{}}
	hasToolCall := false
	for _, item := range r.Output {
		blocks := convertOutputItem(item)
		for _, b := range blocks {
			if b.Type == llm.ContentToolCall {
				hasToolCall = true
			}
		}
		out.Content = append(out.Content, blocks...)
	}
	if r.Usage != nil {
		out.Usage = llm.Usage{
			InputTokens:     r.Usage.InputTokens,
			OutputTokens:    r.Usage.OutputTokens,
			CacheReadTokens: r.Usage.InputTokensDetails.CachedTokens,
			ReasoningTokens: r.Usage.OutputTokensDetails.ReasoningTokens,
		}
	}
	out.ProviderStopReason = r.Status
	switch {
	case r.Status == "incomplete" && r.IncompleteDetails != nil:
		out.ProviderStopReason = r.IncompleteDetails.Reason
		switch r.IncompleteDetails.Reason {
		case "max_output_tokens":
			out.StopReason = llm.StopMaxTokens
		case "content_filter":
			out.StopReason = llm.StopContentFilter
		default:
			out.StopReason = llm.StopOther
		}
	case hasToolCall:
		out.StopReason = llm.StopToolUse
	case r.Status == "completed" || r.Status == "":
		out.StopReason = llm.StopEndTurn
	default:
		out.StopReason = llm.StopOther
	}
	return out, nil
}

// convertOutputItem maps one output item to content blocks. A message item
// yields one text block per content part; unknown item types (built-in tool
// calls etc.) are skipped.
func convertOutputItem(item outputItem) []llm.ContentBlock {
	switch item.Type {
	case "message":
		var blocks []llm.ContentBlock
		for _, c := range item.Content {
			switch c.Type {
			case "output_text":
				blocks = append(blocks, llm.TextBlock(c.Text))
			case "refusal":
				blocks = append(blocks, llm.TextBlock(c.Refusal))
			}
		}
		return blocks
	case "function_call":
		args := item.Arguments
		if args == "" {
			args = "{}"
		}
		return []llm.ContentBlock{llm.ToolCallBlock(llm.ToolCall{
			ID: item.CallID, Name: item.Name, Arguments: json.RawMessage(args),
		})}
	case "reasoning":
		return []llm.ContentBlock{reasoningBlock(item)}
	}
	return nil
}

func reasoningBlock(item outputItem) llm.ContentBlock {
	parts := make([]string, 0, len(item.Summary))
	for _, s := range item.Summary {
		parts = append(parts, s.Text)
	}
	return llm.ContentBlock{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{
		Provider:  ProviderName,
		ID:        item.ID,
		Text:      strings.Join(parts, "\n\n"),
		Encrypted: item.EncryptedContent,
	}}
}

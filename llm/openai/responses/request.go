package responses

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/EvanFromDowntown/Dio/internal/transport"
	"github.com/EvanFromDowntown/Dio/llm"
)

type apiRequest struct {
	Model           string        `json:"model"`
	Instructions    string        `json:"instructions,omitempty"`
	Input           []any         `json:"input"`
	Tools           []apiTool     `json:"tools,omitempty"`
	ToolChoice      any           `json:"tool_choice,omitempty"`
	MaxOutputTokens int           `json:"max_output_tokens,omitempty"`
	Temperature     *float64      `json:"temperature,omitempty"`
	TopP            *float64      `json:"top_p,omitempty"`
	Reasoning       *apiReasoning `json:"reasoning,omitempty"`
	Stream          bool          `json:"stream,omitempty"`
}

type apiTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      bool            `json:"strict"`
}

type apiReasoning struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

type messageItem struct {
	Type    string         `json:"type"`
	Role    string         `json:"role"`
	Content []contentInput `json:"content"`
}

type contentInput struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

type functionCallItem struct {
	Type      string `json:"type"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type functionCallOutputItem struct {
	Type   string `json:"type"`
	CallID string `json:"call_id"`
	Output string `json:"output"`
}

type reasoningItem struct {
	Type             string        `json:"type"`
	ID               string        `json:"id"`
	Summary          []summaryPart `json:"summary"`
	EncryptedContent string        `json:"encrypted_content,omitempty"`
}

type summaryPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

var emptyObjectSchema = json.RawMessage(`{"type":"object","properties":{}}`)

func encodeRequest(req *llm.Request, stream bool) ([]byte, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	if len(req.StopSequences) > 0 {
		return nil, errors.New("openai: the Responses API does not support stop sequences")
	}
	input, err := encodeInput(req.Messages)
	if err != nil {
		return nil, err
	}
	out := apiRequest{
		Model:           req.Model,
		Instructions:    req.System,
		Input:           input,
		MaxOutputTokens: req.MaxTokens,
		Temperature:     req.Temperature,
		TopP:            req.TopP,
		Stream:          stream,
	}
	for _, t := range req.Tools {
		params := t.InputSchema
		if len(params) == 0 {
			params = emptyObjectSchema
		}
		out.Tools = append(out.Tools, apiTool{
			Type:        "function",
			Name:        t.Name,
			Description: t.Description,
			Parameters:  params,
		})
	}
	if tc := req.ToolChoice; tc != nil {
		switch tc.Mode {
		case llm.ToolChoiceAuto, llm.ToolChoiceNone, llm.ToolChoiceRequired:
			out.ToolChoice = string(tc.Mode)
		case llm.ToolChoiceTool:
			out.ToolChoice = map[string]string{"type": "function", "name": tc.Name}
		default:
			return nil, fmt.Errorf("openai: unknown tool choice mode %q", tc.Mode)
		}
	}
	if r := req.Reasoning; r != nil {
		out.Reasoning = &apiReasoning{Effort: r.Effort, Summary: r.Summary}
	}
	return transport.MarshalWithExtra(out, req.Extra)
}

// encodeInput flattens messages into Responses input items. Text and images
// stay grouped in message items; tool calls, tool results and reasoning
// become standalone items in their original order.
func encodeInput(msgs []llm.Message) ([]any, error) {
	var items []any
	for i, m := range msgs {
		var pending []contentInput
		flush := func() {
			if len(pending) > 0 {
				items = append(items, messageItem{Type: "message", Role: string(m.Role), Content: pending})
				pending = nil
			}
		}
		for _, b := range m.Content {
			switch b.Type {
			case llm.ContentText:
				typ := "input_text"
				if m.Role == llm.RoleAssistant {
					typ = "output_text"
				}
				pending = append(pending, contentInput{Type: typ, Text: b.Text})
			case llm.ContentImage:
				if m.Role != llm.RoleUser {
					return nil, fmt.Errorf("openai: message %d: images are only supported in user messages", i)
				}
				url, err := imageURL(b.Image)
				if err != nil {
					return nil, fmt.Errorf("openai: message %d: %w", i, err)
				}
				pending = append(pending, contentInput{Type: "input_image", ImageURL: url})
			case llm.ContentToolCall:
				if b.ToolCall == nil {
					return nil, fmt.Errorf("openai: message %d: tool_call block without payload", i)
				}
				flush()
				args := string(b.ToolCall.Arguments)
				if args == "" {
					args = "{}"
				}
				items = append(items, functionCallItem{
					Type: "function_call", CallID: b.ToolCall.ID, Name: b.ToolCall.Name, Arguments: args,
				})
			case llm.ContentToolResult:
				if b.ToolResult == nil {
					return nil, fmt.Errorf("openai: message %d: tool_result block without payload", i)
				}
				flush()
				items = append(items, functionCallOutputItem{
					Type: "function_call_output", CallID: b.ToolResult.ToolCallID, Output: b.ToolResult.Content,
				})
			case llm.ContentReasoning:
				r := b.Reasoning
				if r == nil || llm.NormalizeProvider(r.Provider) != ProviderName || r.ID == "" {
					continue
				}
				flush()
				summary := []summaryPart{}
				if r.Text != "" {
					summary = append(summary, summaryPart{Type: "summary_text", Text: r.Text})
				}
				items = append(items, reasoningItem{
					Type: "reasoning", ID: r.ID, Summary: summary, EncryptedContent: r.Encrypted,
				})
			default:
				return nil, fmt.Errorf("openai: message %d: unsupported content type %q", i, b.Type)
			}
		}
		flush()
	}
	return items, nil
}

func imageURL(img *llm.Image) (string, error) {
	switch {
	case img == nil:
		return "", errors.New("image block without payload")
	case img.URL != "":
		return img.URL, nil
	case img.Data != "" && img.MediaType != "":
		return "data:" + img.MediaType + ";base64," + img.Data, nil
	default:
		return "", errors.New("image needs a URL or base64 data with a media type")
	}
}

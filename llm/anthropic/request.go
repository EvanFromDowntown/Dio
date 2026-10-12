package anthropic

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/EvanFromDowntown/Dio/internal/transport"
	"github.com/EvanFromDowntown/Dio/llm"
)

type apiRequest struct {
	Model         string         `json:"model"`
	MaxTokens     int            `json:"max_tokens"`
	System        string         `json:"system,omitempty"`
	Messages      []apiMessage   `json:"messages"`
	Tools         []apiTool      `json:"tools,omitempty"`
	ToolChoice    *apiToolChoice `json:"tool_choice,omitempty"`
	Temperature   *float64       `json:"temperature,omitempty"`
	TopP          *float64       `json:"top_p,omitempty"`
	StopSequences []string       `json:"stop_sequences,omitempty"`
	Thinking      *apiThinking   `json:"thinking,omitempty"`
	Stream        bool           `json:"stream,omitempty"`
}

type apiMessage struct {
	Role    string `json:"role"`
	Content []any  `json:"content"`
}

type apiTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type apiToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

type apiThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens"`
}

type textBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type imageBlock struct {
	Type   string      `json:"type"`
	Source imageSource `json:"source"`
}

type imageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type toolUseBlock struct {
	Type  string          `json:"type"`
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

type toolResultBlock struct {
	Type      string `json:"type"`
	ToolUseID string `json:"tool_use_id"`
	Content   string `json:"content"`
	IsError   bool   `json:"is_error,omitempty"`
}

type thinkingBlock struct {
	Type      string `json:"type"`
	Thinking  string `json:"thinking"`
	Signature string `json:"signature"`
}

type redactedThinkingBlock struct {
	Type string `json:"type"`
	Data string `json:"data"`
}

var emptyObjectSchema = json.RawMessage(`{"type":"object","properties":{}}`)

// effortBudgets maps ReasoningConfig.Effort to a thinking budget when
// BudgetTokens is not set.
var effortBudgets = map[string]int{"low": 1024, "medium": 4096, "high": 16384}

func encodeRequest(req *llm.Request, defaultMaxTokens int, stream bool) ([]byte, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	out := apiRequest{
		Model:         req.Model,
		MaxTokens:     req.MaxTokens,
		System:        req.System,
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		StopSequences: req.StopSequences,
		Stream:        stream,
	}
	if out.MaxTokens == 0 {
		out.MaxTokens = defaultMaxTokens
	}
	for i, m := range req.Messages {
		msg, err := encodeMessage(m)
		if err != nil {
			return nil, fmt.Errorf("anthropic: message %d: %w", i, err)
		}
		if len(msg.Content) > 0 {
			out.Messages = append(out.Messages, msg)
		}
	}
	for _, t := range req.Tools {
		schema := t.InputSchema
		if len(schema) == 0 {
			schema = emptyObjectSchema
		}
		out.Tools = append(out.Tools, apiTool{Name: t.Name, Description: t.Description, InputSchema: schema})
	}
	if tc := req.ToolChoice; tc != nil {
		switch tc.Mode {
		case llm.ToolChoiceAuto:
			out.ToolChoice = &apiToolChoice{Type: "auto"}
		case llm.ToolChoiceNone:
			out.ToolChoice = &apiToolChoice{Type: "none"}
		case llm.ToolChoiceRequired:
			out.ToolChoice = &apiToolChoice{Type: "any"}
		case llm.ToolChoiceTool:
			out.ToolChoice = &apiToolChoice{Type: "tool", Name: tc.Name}
		default:
			return nil, fmt.Errorf("anthropic: unknown tool choice mode %q", tc.Mode)
		}
	}
	if r := req.Reasoning; r != nil {
		budget := r.BudgetTokens
		if budget == 0 {
			budget = effortBudgets[r.Effort]
		}
		if budget == 0 {
			return nil, fmt.Errorf("anthropic: reasoning needs budget_tokens or a known effort, got %q", r.Effort)
		}
		out.Thinking = &apiThinking{Type: "enabled", BudgetTokens: budget}
	}
	return transport.MarshalWithExtra(out, req.Extra)
}

func encodeMessage(m llm.Message) (apiMessage, error) {
	msg := apiMessage{Role: string(m.Role), Content: []any{}}
	for _, b := range m.Content {
		switch b.Type {
		case llm.ContentText:
			msg.Content = append(msg.Content, textBlock{Type: "text", Text: b.Text})
		case llm.ContentImage:
			src, err := encodeImage(b.Image)
			if err != nil {
				return msg, err
			}
			msg.Content = append(msg.Content, imageBlock{Type: "image", Source: src})
		case llm.ContentToolCall:
			if b.ToolCall == nil {
				return msg, errors.New("tool_call block without payload")
			}
			input := b.ToolCall.Arguments
			if len(input) == 0 {
				input = json.RawMessage(`{}`)
			}
			msg.Content = append(msg.Content, toolUseBlock{
				Type: "tool_use", ID: b.ToolCall.ID, Name: b.ToolCall.Name, Input: input,
			})
		case llm.ContentToolResult:
			if b.ToolResult == nil {
				return msg, errors.New("tool_result block without payload")
			}
			msg.Content = append(msg.Content, toolResultBlock{
				Type: "tool_result", ToolUseID: b.ToolResult.ToolCallID,
				Content: b.ToolResult.Content, IsError: b.ToolResult.IsError,
			})
		case llm.ContentReasoning:
			r := b.Reasoning
			if r == nil || r.Provider != ProviderName {
				continue
			}
			if r.Encrypted != "" && r.Signature == "" {
				msg.Content = append(msg.Content, redactedThinkingBlock{Type: "redacted_thinking", Data: r.Encrypted})
			} else {
				msg.Content = append(msg.Content, thinkingBlock{Type: "thinking", Thinking: r.Text, Signature: r.Signature})
			}
		default:
			return msg, fmt.Errorf("unsupported content type %q", b.Type)
		}
	}
	return msg, nil
}

func encodeImage(img *llm.Image) (imageSource, error) {
	switch {
	case img == nil:
		return imageSource{}, errors.New("image block without payload")
	case img.URL != "":
		return imageSource{Type: "url", URL: img.URL}, nil
	case img.Data != "" && img.MediaType != "":
		return imageSource{Type: "base64", MediaType: img.MediaType, Data: img.Data}, nil
	default:
		return imageSource{}, errors.New("image needs a URL or base64 data with a media type")
	}
}

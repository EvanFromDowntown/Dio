package chatcompletions

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/EvanFromDowntown/Dio/internal/transport"
	"github.com/EvanFromDowntown/Dio/llm"
)

type apiMessage struct {
	Role string `json:"role"`
	// Content is a string, an array of parts, or null (assistant messages
	// that only call tools).
	Content          any           `json:"content"`
	ReasoningContent string        `json:"reasoning_content,omitempty"`
	ToolCalls        []apiToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string        `json:"tool_call_id,omitempty"`
}

type contentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

type imageURL struct {
	URL string `json:"url"`
}

type apiToolCall struct {
	Index    *int         `json:"index,omitempty"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function functionCall `json:"function"`
}

type functionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`
}

type apiTool struct {
	Type     string      `json:"type"`
	Function apiFunction `json:"function"`
}

type apiFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

var emptyObjectSchema = json.RawMessage(`{"type":"object","properties":{}}`)

func (c *Client) encodeRequest(req *llm.Request, stream bool) ([]byte, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}
	body := map[string]any{"model": req.Model}

	var msgs []apiMessage
	if req.System != "" {
		msgs = append(msgs, apiMessage{Role: "system", Content: req.System})
	}
	for i, m := range req.Messages {
		encoded, err := c.encodeMessage(m)
		if err != nil {
			return nil, fmt.Errorf("chatcompletions: message %d: %w", i, err)
		}
		msgs = append(msgs, encoded...)
	}
	body["messages"] = msgs

	if len(req.Tools) > 0 {
		tools := make([]apiTool, len(req.Tools))
		for i, t := range req.Tools {
			params := t.InputSchema
			if len(params) == 0 {
				params = emptyObjectSchema
			}
			tools[i] = apiTool{Type: "function", Function: apiFunction{Name: t.Name, Description: t.Description, Parameters: params}}
		}
		body["tools"] = tools
	}
	if tc := req.ToolChoice; tc != nil {
		switch tc.Mode {
		case llm.ToolChoiceAuto, llm.ToolChoiceNone, llm.ToolChoiceRequired:
			body["tool_choice"] = string(tc.Mode)
		case llm.ToolChoiceTool:
			body["tool_choice"] = map[string]any{"type": "function", "function": map[string]string{"name": tc.Name}}
		default:
			return nil, fmt.Errorf("chatcompletions: unknown tool choice mode %q", tc.Mode)
		}
	}
	if req.MaxTokens > 0 {
		body[c.maxTokensField] = req.MaxTokens
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if len(req.StopSequences) > 0 {
		body["stop"] = req.StopSequences
	}
	if r := req.Reasoning; r != nil && r.Effort != "" {
		body["reasoning_effort"] = r.Effort
	}
	if stream {
		body["stream"] = true
		if c.streamUsage {
			body["stream_options"] = map[string]bool{"include_usage": true}
		}
	}
	return transport.MarshalWithExtra(body, req.Extra)
}

// encodeMessage maps one message to one or more wire messages: tool results
// become separate "tool" messages, in their original order relative to any
// user text around them.
func (c *Client) encodeMessage(m llm.Message) ([]apiMessage, error) {
	if m.Role == llm.RoleAssistant {
		msg, err := c.encodeAssistant(m)
		return []apiMessage{msg}, err
	}
	var (
		out     []apiMessage
		pending []contentPart
		images  bool
	)
	flush := func() {
		if len(pending) == 0 {
			return
		}
		if images {
			out = append(out, apiMessage{Role: "user", Content: pending})
		} else {
			texts := make([]string, len(pending))
			for i, p := range pending {
				texts[i] = p.Text
			}
			out = append(out, apiMessage{Role: "user", Content: strings.Join(texts, "")})
		}
		pending, images = nil, false
	}
	for _, b := range m.Content {
		switch b.Type {
		case llm.ContentText:
			pending = append(pending, contentPart{Type: "text", Text: b.Text})
		case llm.ContentImage:
			url, err := imageDataURL(b.Image)
			if err != nil {
				return nil, err
			}
			pending = append(pending, contentPart{Type: "image_url", ImageURL: &imageURL{URL: url}})
			images = true
		case llm.ContentToolResult:
			if b.ToolResult == nil {
				return nil, errors.New("tool_result block without payload")
			}
			flush()
			content := b.ToolResult.Content
			if b.ToolResult.IsError {
				content = "Error: " + content
			}
			out = append(out, apiMessage{Role: "tool", ToolCallID: b.ToolResult.ToolCallID, Content: content})
		case llm.ContentReasoning:
		default:
			return nil, fmt.Errorf("unsupported content type %q in user message", b.Type)
		}
	}
	flush()
	return out, nil
}

func (c *Client) encodeAssistant(m llm.Message) (apiMessage, error) {
	msg := apiMessage{Role: "assistant"}
	var text, reasoning strings.Builder
	for _, b := range m.Content {
		switch b.Type {
		case llm.ContentText:
			text.WriteString(b.Text)
		case llm.ContentToolCall:
			if b.ToolCall == nil {
				return msg, errors.New("tool_call block without payload")
			}
			args := string(b.ToolCall.Arguments)
			if args == "" {
				args = "{}"
			}
			msg.ToolCalls = append(msg.ToolCalls, apiToolCall{
				ID: b.ToolCall.ID, Type: "function",
				Function: functionCall{Name: b.ToolCall.Name, Arguments: args},
			})
		case llm.ContentReasoning:
			if r := b.Reasoning; c.replayReasoning && r != nil && llm.NormalizeProvider(r.Provider) == ProviderName {
				reasoning.WriteString(r.Text)
			}
		default:
			return msg, fmt.Errorf("unsupported content type %q in assistant message", b.Type)
		}
	}
	if text.Len() > 0 || len(msg.ToolCalls) == 0 {
		msg.Content = text.String()
	}
	msg.ReasoningContent = reasoning.String()
	return msg, nil
}

func imageDataURL(img *llm.Image) (string, error) {
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

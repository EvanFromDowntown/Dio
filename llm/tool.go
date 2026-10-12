package llm

import "encoding/json"

// Tool declares a function the model may call.
type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	// InputSchema is a JSON Schema object describing the arguments. An empty
	// schema is sent as an object with no properties.
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
}

// ToolCall is a model request to invoke a tool.
type ToolCall struct {
	// ID correlates the call with its ToolResult (OpenAI call_id, Anthropic
	// tool_use id).
	ID   string `json:"id"`
	Name string `json:"name"`
	// Arguments is the JSON-encoded argument object produced by the model.
	Arguments json.RawMessage `json:"arguments"`
}

type toolCallJSON struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// MarshalJSON stores Arguments as a JSON string holding the exact bytes the
// model produced, so persisted calls round-trip losslessly even when the
// arguments are not valid JSON.
func (c ToolCall) MarshalJSON() ([]byte, error) {
	args, err := json.Marshal(string(c.Arguments))
	if err != nil {
		return nil, err
	}
	return json.Marshal(toolCallJSON{ID: c.ID, Name: c.Name, Arguments: args})
}

// UnmarshalJSON accepts arguments as a string (MarshalJSON's form) or as an
// inline JSON value.
func (c *ToolCall) UnmarshalJSON(data []byte) error {
	var raw toolCallJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	c.ID, c.Name, c.Arguments = raw.ID, raw.Name, nil
	switch {
	case len(raw.Arguments) == 0 || string(raw.Arguments) == "null":
	case raw.Arguments[0] == '"':
		var s string
		if err := json.Unmarshal(raw.Arguments, &s); err != nil {
			return err
		}
		if s != "" {
			c.Arguments = json.RawMessage(s)
		}
	default:
		c.Arguments = append(json.RawMessage(nil), raw.Arguments...)
	}
	return nil
}

// ToolResult is the outcome of executing a ToolCall.
type ToolResult struct {
	ToolCallID string `json:"tool_call_id"`
	Content    string `json:"content"`
	IsError    bool   `json:"is_error,omitempty"`
}

type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	// ToolChoiceTool forces the tool named by ToolChoice.Name.
	ToolChoiceTool ToolChoiceMode = "tool"
)

// ToolChoice controls whether and which tools the model must call.
type ToolChoice struct {
	Mode ToolChoiceMode `json:"mode"`
	Name string         `json:"name,omitempty"`
}

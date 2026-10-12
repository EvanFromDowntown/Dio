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

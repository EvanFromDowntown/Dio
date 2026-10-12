// Package tool defines executable capabilities exposed to the model and a
// registry that converts them into llm.Tool declarations.
package tool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/EvanFromDowntown/Dio/llm"
)

// Tool is a capability the model can invoke.
type Tool interface {
	Name() string
	Description() string
	// Schema is the JSON Schema of the argument object; nil means no
	// arguments.
	Schema() json.RawMessage
	// Execute runs the tool with the model-provided JSON arguments. The
	// returned string is sent back to the model; a non-nil error is sent back
	// as an error result instead of aborting the agent loop.
	Execute(ctx context.Context, args json.RawMessage) (string, error)
}

// Func adapts a function to Tool.
type Func struct {
	ToolName        string
	ToolDescription string
	InputSchema     json.RawMessage
	Fn              func(ctx context.Context, args json.RawMessage) (string, error)
}

var _ Tool = (*Func)(nil)

func (f *Func) Name() string            { return f.ToolName }
func (f *Func) Description() string     { return f.ToolDescription }
func (f *Func) Schema() json.RawMessage { return f.InputSchema }
func (f *Func) Execute(ctx context.Context, args json.RawMessage) (string, error) {
	return f.Fn(ctx, args)
}

// Registry holds uniquely named tools in registration order. It is not safe
// for concurrent mutation; register everything before use.
type Registry struct {
	order  []Tool
	byName map[string]Tool
}

func NewRegistry() *Registry {
	return &Registry{byName: map[string]Tool{}}
}

// Register adds tools, rejecting empty or duplicate names.
func (r *Registry) Register(tools ...Tool) error {
	for _, t := range tools {
		if t == nil {
			return errors.New("tool: nil tool")
		}
		name := t.Name()
		if name == "" {
			return errors.New("tool: empty tool name")
		}
		if _, dup := r.byName[name]; dup {
			return fmt.Errorf("tool: duplicate tool name %q", name)
		}
		r.byName[name] = t
		r.order = append(r.order, t)
	}
	return nil
}

func (r *Registry) Get(name string) (Tool, bool) {
	t, ok := r.byName[name]
	return t, ok
}

// Tools returns all tools in registration order.
func (r *Registry) Tools() []Tool {
	return append([]Tool(nil), r.order...)
}

func (r *Registry) Len() int { return len(r.order) }

// LLMTools converts the registry into request tool declarations.
func (r *Registry) LLMTools() []llm.Tool {
	if len(r.order) == 0 {
		return nil
	}
	out := make([]llm.Tool, len(r.order))
	for i, t := range r.order {
		out[i] = llm.Tool{Name: t.Name(), Description: t.Description(), InputSchema: t.Schema()}
	}
	return out
}

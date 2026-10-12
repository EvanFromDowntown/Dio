package agent

import (
	"context"
	"encoding/json"
	"log/slog"

	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/tool"
)

// Plugin packages tools, prompt fragments and hooks. Capabilities are
// discovered by type assertion against ToolProvider, PromptProvider,
// ModelHook, ToolHook and FinishHook. Hooks run in plugin registration order.
type Plugin interface {
	Name() string
	// Init is called once by New, in registration order, before any
	// capability is collected.
	Init(ctx context.Context, host Host) error
}

// Host is the agent-side API available to plugins.
type Host interface {
	// Logger is scoped to the plugin (it carries a "plugin" attribute).
	Logger() *slog.Logger
}

// ToolProvider contributes tools. Tools is called once, after Init.
type ToolProvider interface {
	Tools() []tool.Tool
}

// PromptProvider contributes a system prompt fragment. It is called at the
// start of every run; an empty fragment is skipped.
type PromptProvider interface {
	SystemPrompt(ctx context.Context) (string, error)
}

// ModelHook observes or rewrites each model call. BeforeModel may mutate the
// request (it is rebuilt from the history every turn, so changes do not
// persist); AfterModel may mutate the response before it is appended to the
// history. A returned error aborts the run.
type ModelHook interface {
	BeforeModel(ctx context.Context, req *llm.Request) error
	AfterModel(ctx context.Context, resp *llm.Response) error
}

// ToolHook guards each tool call. BeforeTool hooks run in order until one
// denies; AfterTool hooks run for every result, including denied and failed
// calls, and may rewrite it. A returned error aborts the run.
type ToolHook interface {
	BeforeTool(ctx context.Context, call *llm.ToolCall) (Decision, error)
	AfterTool(ctx context.Context, call llm.ToolCall, result *llm.ToolResult) error
}

// FinishHook is notified exactly once when a run ends, whatever the outcome.
// It receives a context that is not canceled with the run's context.
type FinishHook interface {
	OnFinish(ctx context.Context, result *Result) error
}

type Action int

const (
	ActionAllow Action = iota
	ActionDeny
	ActionModify
)

func (a Action) String() string {
	switch a {
	case ActionAllow:
		return "allow"
	case ActionDeny:
		return "deny"
	case ActionModify:
		return "modify"
	}
	return "unknown"
}

// Decision is a BeforeTool verdict. The zero value allows the call.
type Decision struct {
	Action Action
	// Reason is returned to the model as an error result when denying.
	Reason string
	// Arguments replace the call arguments when modifying; later hooks and
	// the tool see the new arguments.
	Arguments json.RawMessage
}

func Allow() Decision { return Decision{Action: ActionAllow} }

func Deny(reason string) Decision { return Decision{Action: ActionDeny, Reason: reason} }

func ModifyArguments(args json.RawMessage) Decision {
	return Decision{Action: ActionModify, Arguments: args}
}

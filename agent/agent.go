// Package agent implements the tool-calling loop on top of llm.Client.
// Capabilities are tools, constraints are hooks, and plugins only package
// them together.
package agent

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"slices"
	"strings"

	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/tool"
)

// DefaultMaxTurns bounds model calls per run when Config.MaxTurns is zero.
const DefaultMaxTurns = 25

var (
	// ErrMaxTurns is returned when the run hits Config.MaxTurns while the
	// model still requests tools.
	ErrMaxTurns = errors.New("agent: max turns reached")
	// ErrStopped is the Result.Err seen by FinishHooks when a Stream consumer
	// stops iterating early.
	ErrStopped = errors.New("agent: stream consumer stopped")
)

type Config struct {
	Client llm.Client
	Model  string
	// System is the base system prompt; plugin fragments are appended to it.
	System      string
	MaxTokens   int
	Temperature *float64
	Reasoning   *llm.ReasoningConfig
	// MaxTurns caps model calls per run. Zero means DefaultMaxTurns.
	MaxTurns int
	// Tools are registered before plugin tools.
	Tools   []tool.Tool
	Plugins []Plugin
	Logger  *slog.Logger
}

type Agent struct {
	cfg          Config
	logger       *slog.Logger
	tools        *tool.Registry
	prompts      []PromptProvider
	modelHooks   []ModelHook
	toolHooks    []ToolHook
	messageHooks []MessageHook
	finishHooks  []FinishHook
}

type host struct{ logger *slog.Logger }

func (h host) Logger() *slog.Logger { return h.logger }

// New initializes plugins in order and collects their capabilities.
func New(ctx context.Context, cfg Config) (*Agent, error) {
	if cfg.Client == nil {
		return nil, errors.New("agent: config needs a Client")
	}
	if cfg.Model == "" {
		return nil, errors.New("agent: config needs a Model")
	}
	if cfg.MaxTurns <= 0 {
		cfg.MaxTurns = DefaultMaxTurns
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	a := &Agent{cfg: cfg, logger: logger, tools: tool.NewRegistry()}
	if err := a.tools.Register(cfg.Tools...); err != nil {
		return nil, fmt.Errorf("agent: %w", err)
	}

	seen := map[string]bool{}
	for _, p := range cfg.Plugins {
		if p == nil {
			return nil, errors.New("agent: nil plugin")
		}
		name := p.Name()
		if seen[name] {
			return nil, fmt.Errorf("agent: duplicate plugin %q", name)
		}
		seen[name] = true
		if err := p.Init(ctx, host{logger: logger.With("plugin", name)}); err != nil {
			return nil, fmt.Errorf("agent: init plugin %q: %w", name, err)
		}
	}
	for _, p := range cfg.Plugins {
		if tp, ok := p.(ToolProvider); ok {
			if err := a.tools.Register(tp.Tools()...); err != nil {
				return nil, fmt.Errorf("agent: plugin %q: %w", p.Name(), err)
			}
		}
		if pp, ok := p.(PromptProvider); ok {
			a.prompts = append(a.prompts, pp)
		}
		if h, ok := p.(ModelHook); ok {
			a.modelHooks = append(a.modelHooks, h)
		}
		if h, ok := p.(ToolHook); ok {
			a.toolHooks = append(a.toolHooks, h)
		}
		if h, ok := p.(MessageHook); ok {
			a.messageHooks = append(a.messageHooks, h)
		}
		if h, ok := p.(FinishHook); ok {
			a.finishHooks = append(a.finishHooks, h)
		}
	}
	return a, nil
}

// Tools returns the registered tools in registration order.
func (a *Agent) Tools() []tool.Tool { return a.tools.Tools() }

type FinishReason string

const (
	// FinishCompleted: the model answered without requesting tools.
	FinishCompleted FinishReason = "completed"
	FinishMaxTurns  FinishReason = "max_turns"
	// FinishCanceled: the context was canceled or a Stream consumer stopped.
	FinishCanceled FinishReason = "canceled"
	FinishError    FinishReason = "error"
)

// Result is the outcome of a run. It is returned even when the run fails, so
// the partial history can be inspected or resumed.
type Result struct {
	// Messages is the input history followed by every message the run
	// appended. If the run stops between a tool request and its results,
	// the missing results are filled with error results, so the history
	// stays valid for a follow-up run.
	Messages []llm.Message
	// Response is the last model response, if any.
	Response     *llm.Response
	Turns        int
	Usage        llm.Usage
	FinishReason FinishReason
	Err          error
}

// Text returns the text of the last model response.
func (r *Result) Text() string {
	if r.Response == nil {
		return ""
	}
	return r.Response.Text()
}

type EventType string

const (
	// EventTurnStart precedes each model call.
	EventTurnStart EventType = "turn_start"
	// EventModel wraps a streaming event from the model (LLM field).
	EventModel EventType = "model"
	// EventModelResponse carries the response after AfterModel hooks.
	EventModelResponse EventType = "model_response"
	// EventToolCall carries the call after BeforeTool hooks, before execution.
	EventToolCall EventType = "tool_call"
	// EventToolResult carries the result after AfterTool hooks.
	EventToolResult EventType = "tool_result"
	// EventDone is always last and carries the Result.
	EventDone EventType = "done"
)

type Event struct {
	Type       EventType
	Turn       int
	LLM        *llm.StreamEvent
	Response   *llm.Response
	ToolCall   *llm.ToolCall
	ToolResult *llm.ToolResult
	Result     *Result
}

// Run executes the loop with non-streaming model calls.
func (a *Agent) Run(ctx context.Context, messages []llm.Message) (*Result, error) {
	return a.run(ctx, messages, nil)
}

// Stream executes the loop with streaming model calls and yields events as
// they happen. The sequence always ends with an EventDone whose error is the
// run's error. Breaking out of the loop stops the run.
func (a *Agent) Stream(ctx context.Context, messages []llm.Message) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		stopped := false
		res, err := a.run(ctx, messages, func(ev Event) bool {
			if !yield(ev, nil) {
				stopped = true
				return false
			}
			return true
		})
		if stopped {
			return
		}
		yield(Event{Type: EventDone, Turn: res.Turns, Result: res}, err)
	}
}

type emitFunc func(Event) bool

func (a *Agent) run(ctx context.Context, input []llm.Message, emit emitFunc) (res *Result, err error) {
	res = &Result{Messages: slices.Clone(input)}
	defer func() { err = a.finish(ctx, res, err) }()
	send := func(ev Event) error {
		if emit != nil && !emit(ev) {
			return ErrStopped
		}
		return nil
	}

	system, err := a.systemPrompt(ctx)
	if err != nil {
		return res, err
	}
	for turn := 1; ; turn++ {
		if turn > a.cfg.MaxTurns {
			return res, ErrMaxTurns
		}
		if err := ctx.Err(); err != nil {
			return res, err
		}
		res.Turns = turn
		if err := send(Event{Type: EventTurnStart, Turn: turn}); err != nil {
			return res, err
		}

		req := a.buildRequest(system, res.Messages)
		for _, h := range a.modelHooks {
			if err := h.BeforeModel(ctx, req); err != nil {
				return res, fmt.Errorf("agent: BeforeModel hook: %w", err)
			}
		}
		resp, err := a.callModel(ctx, req, turn, emit)
		if err != nil {
			return res, err
		}
		for _, h := range a.modelHooks {
			if err := h.AfterModel(ctx, resp); err != nil {
				return res, fmt.Errorf("agent: AfterModel hook: %w", err)
			}
		}
		res.Response = resp
		addUsage(&res.Usage, resp.Usage)
		if err := a.appendMessage(ctx, res, resp.Message()); err != nil {
			return res, err
		}
		if err := send(Event{Type: EventModelResponse, Turn: turn, Response: resp}); err != nil {
			return res, err
		}

		calls := resp.ToolCalls()
		if len(calls) == 0 {
			return res, nil
		}
		results, err := a.runTools(ctx, turn, calls, send)
		if herr := a.appendMessage(ctx, res, llm.ToolResults(results...)); err == nil {
			err = herr
		}
		if err != nil {
			return res, err
		}
	}
}

// appendMessage is the only place the run grows the history, so MessageHooks
// see exactly what a follow-up run will receive.
func (a *Agent) appendMessage(ctx context.Context, res *Result, msg llm.Message) error {
	res.Messages = append(res.Messages, msg)
	hookCtx := context.WithoutCancel(ctx)
	for _, h := range a.messageHooks {
		if err := h.OnMessage(hookCtx, msg); err != nil {
			return fmt.Errorf("agent: OnMessage hook: %w", err)
		}
	}
	return nil
}

// runTools executes calls sequentially. It always returns one result per
// call; calls not reached because of an error get an error result.
func (a *Agent) runTools(ctx context.Context, turn int, calls []llm.ToolCall, send func(Event) error) ([]llm.ToolResult, error) {
	results := make([]llm.ToolResult, 0, len(calls))
	var err error
	for _, call := range calls {
		if err = ctx.Err(); err != nil {
			break
		}
		var r llm.ToolResult
		r, err = a.runTool(ctx, turn, call, send)
		if err != nil {
			break
		}
		results = append(results, r)
	}
	for _, call := range calls[len(results):] {
		results = append(results, llm.ToolResult{
			ToolCallID: call.ID, Content: "Tool call was interrupted before a result was recorded.", IsError: true,
		})
	}
	return results, err
}

func (a *Agent) runTool(ctx context.Context, turn int, call llm.ToolCall, send func(Event) error) (llm.ToolResult, error) {
	id := call.ID
	result := llm.ToolResult{ToolCallID: id}
	denied := false
	for _, h := range a.toolHooks {
		d, err := h.BeforeTool(ctx, &call)
		if err != nil {
			return result, fmt.Errorf("agent: BeforeTool hook for %q: %w", call.Name, err)
		}
		call.ID = id
		switch d.Action {
		case ActionAllow:
		case ActionModify:
			call.Arguments = d.Arguments
		case ActionDeny:
			denied = true
			result.IsError = true
			result.Content = "Tool call denied."
			if d.Reason != "" {
				result.Content = "Tool call denied: " + d.Reason
			}
		default:
			return result, fmt.Errorf("agent: BeforeTool hook for %q returned unknown action %d", call.Name, d.Action)
		}
		if denied {
			break
		}
	}
	if err := send(Event{Type: EventToolCall, Turn: turn, ToolCall: &call}); err != nil {
		return result, err
	}
	if !denied {
		result = a.execute(ctx, call)
	}
	for _, h := range a.toolHooks {
		if err := h.AfterTool(ctx, call, &result); err != nil {
			return result, fmt.Errorf("agent: AfterTool hook for %q: %w", call.Name, err)
		}
	}
	result.ToolCallID = id
	if err := send(Event{Type: EventToolResult, Turn: turn, ToolCall: &call, ToolResult: &result}); err != nil {
		return result, err
	}
	return result, nil
}

// execute turns every tool failure, including unknown tools and panics, into
// an error result for the model.
func (a *Agent) execute(ctx context.Context, call llm.ToolCall) (result llm.ToolResult) {
	result.ToolCallID = call.ID
	t, ok := a.tools.Get(call.Name)
	if !ok {
		result.IsError = true
		result.Content = fmt.Sprintf("Unknown tool %q.", call.Name)
		return result
	}
	defer func() {
		if p := recover(); p != nil {
			a.logger.Error("tool panicked", "tool", call.Name, "panic", p)
			result.IsError = true
			result.Content = fmt.Sprintf("Tool %q panicked: %v", call.Name, p)
		}
	}()
	out, err := t.Execute(ctx, call.Arguments)
	if err != nil {
		result.IsError = true
		result.Content = err.Error()
		return result
	}
	result.Content = out
	return result
}

func (a *Agent) callModel(ctx context.Context, req *llm.Request, turn int, emit emitFunc) (*llm.Response, error) {
	if emit == nil {
		resp, err := a.cfg.Client.Generate(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("agent: model call (turn %d): %w", turn, err)
		}
		return resp, nil
	}
	s, err := a.cfg.Client.Stream(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("agent: model call (turn %d): %w", turn, err)
	}
	defer s.Close()
	var resp *llm.Response
	for s.Next() {
		ev := s.Event()
		if ev.Type == llm.EventDone {
			resp = ev.Response
		}
		if !emit(Event{Type: EventModel, Turn: turn, LLM: &ev}) {
			return nil, ErrStopped
		}
	}
	if err := s.Err(); err != nil {
		return nil, fmt.Errorf("agent: model stream (turn %d): %w", turn, err)
	}
	if resp == nil {
		return nil, fmt.Errorf("agent: model stream (turn %d): %w", turn, llm.ErrIncompleteStream)
	}
	return resp, nil
}

func (a *Agent) systemPrompt(ctx context.Context) (string, error) {
	var parts []string
	if s := strings.TrimSpace(a.cfg.System); s != "" {
		parts = append(parts, s)
	}
	for _, p := range a.prompts {
		s, err := p.SystemPrompt(ctx)
		if err != nil {
			return "", fmt.Errorf("agent: system prompt: %w", err)
		}
		if s = strings.TrimSpace(s); s != "" {
			parts = append(parts, s)
		}
	}
	return strings.Join(parts, "\n\n"), nil
}

func (a *Agent) buildRequest(system string, history []llm.Message) *llm.Request {
	return &llm.Request{
		Model:       a.cfg.Model,
		System:      system,
		Messages:    slices.Clone(history),
		Tools:       a.tools.LLMTools(),
		MaxTokens:   a.cfg.MaxTokens,
		Temperature: a.cfg.Temperature,
		Reasoning:   a.cfg.Reasoning,
	}
}

func (a *Agent) finish(ctx context.Context, res *Result, err error) error {
	switch {
	case err == nil:
		res.FinishReason = FinishCompleted
	case errors.Is(err, ErrMaxTurns):
		res.FinishReason = FinishMaxTurns
	case errors.Is(err, ErrStopped), errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		res.FinishReason = FinishCanceled
	default:
		res.FinishReason = FinishError
	}
	res.Err = err
	hookCtx := context.WithoutCancel(ctx)
	for _, h := range a.finishHooks {
		if herr := h.OnFinish(hookCtx, res); herr != nil {
			a.logger.Error("OnFinish hook failed", "error", herr)
			if err == nil {
				err = fmt.Errorf("agent: OnFinish hook: %w", herr)
				res.Err = err
			}
		}
	}
	return err
}

func addUsage(dst *llm.Usage, u llm.Usage) {
	dst.InputTokens += u.InputTokens
	dst.OutputTokens += u.OutputTokens
	dst.CacheReadTokens += u.CacheReadTokens
	dst.CacheWriteTokens += u.CacheWriteTokens
	dst.ReasoningTokens += u.ReasoningTokens
}

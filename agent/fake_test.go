package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/EvanFromDowntown/Dio/llm"
)

// step produces the model response for one call.
type step func(ctx context.Context, req *llm.Request) (*llm.Response, error)

func reply(resp *llm.Response) step {
	return func(context.Context, *llm.Request) (*llm.Response, error) { return resp, nil }
}

func textResp(text string) *llm.Response {
	return &llm.Response{
		ID: "resp", Content: []llm.ContentBlock{llm.TextBlock(text)},
		StopReason: llm.StopEndTurn, Usage: llm.Usage{InputTokens: 10, OutputTokens: 1},
	}
}

func toolResp(calls ...llm.ToolCall) *llm.Response {
	r := &llm.Response{ID: "resp", StopReason: llm.StopToolUse, Usage: llm.Usage{InputTokens: 10, OutputTokens: 2}}
	for _, c := range calls {
		r.Content = append(r.Content, llm.ToolCallBlock(c))
	}
	return r
}

func call(id, name, args string) llm.ToolCall {
	return llm.ToolCall{ID: id, Name: name, Arguments: json.RawMessage(args)}
}

// fakeClient replays scripted steps and records every request.
type fakeClient struct {
	mu       sync.Mutex
	steps    []step
	requests []*llm.Request
	streams  []*fakeStream
}

func newFake(steps ...step) *fakeClient { return &fakeClient{steps: steps} }

func (f *fakeClient) next(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	f.mu.Lock()
	snapshot := *req
	snapshot.Messages = slices.Clone(req.Messages)
	f.requests = append(f.requests, &snapshot)
	if len(f.steps) == 0 {
		f.mu.Unlock()
		return nil, errors.New("fake: unexpected model call")
	}
	s := f.steps[0]
	f.steps = f.steps[1:]
	f.mu.Unlock()
	return s(ctx, req)
}

func (f *fakeClient) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	return f.next(ctx, req)
}

func (f *fakeClient) Stream(ctx context.Context, req *llm.Request) (llm.Stream, error) {
	resp, err := f.next(ctx, req)
	if err != nil {
		return nil, err
	}
	s := &fakeStream{events: streamEvents(resp)}
	f.mu.Lock()
	f.streams = append(f.streams, s)
	f.mu.Unlock()
	return s, nil
}

func (f *fakeClient) calls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.requests)
}

// streamEvents renders a response as the unified event sequence.
func streamEvents(resp *llm.Response) []llm.StreamEvent {
	events := []llm.StreamEvent{{Type: llm.EventStart, Response: &llm.Response{ID: resp.ID}}}
	for i, b := range resp.Content {
		b := b
		events = append(events, llm.StreamEvent{Type: llm.EventContentStart, Index: i, Block: &b})
		switch b.Type {
		case llm.ContentText:
			events = append(events, llm.StreamEvent{Type: llm.EventTextDelta, Index: i, Delta: b.Text})
		case llm.ContentToolCall:
			events = append(events, llm.StreamEvent{Type: llm.EventToolCallDelta, Index: i, Delta: string(b.ToolCall.Arguments)})
		}
		events = append(events, llm.StreamEvent{Type: llm.EventContentStop, Index: i, Block: &b})
	}
	return append(events, llm.StreamEvent{Type: llm.EventDone, Response: resp})
}

type fakeStream struct {
	events []llm.StreamEvent
	i      int
	closed bool
}

func (s *fakeStream) Next() bool {
	if s.i >= len(s.events) {
		return false
	}
	s.i++
	return true
}
func (s *fakeStream) Event() llm.StreamEvent { return s.events[s.i-1] }
func (s *fakeStream) Err() error             { return nil }
func (s *fakeStream) Close() error           { s.closed = true; return nil }

// hookPlugin implements every capability through optional funcs and records
// hook invocations into a shared log.
type hookPlugin struct {
	name   string
	log    *[]string
	prompt string
	tools  []toolDef

	beforeModel func(*llm.Request) error
	afterModel  func(*llm.Response) error
	beforeTool  func(*llm.ToolCall) (Decision, error)
	afterTool   func(llm.ToolCall, *llm.ToolResult) error
	onFinish    func(context.Context, *Result) error
	initErr     error
	host        Host
}

type toolDef struct {
	name string
	fn   func(ctx context.Context, args json.RawMessage) (string, error)
}

func (p *hookPlugin) record(format string, args ...any) {
	if p.log != nil {
		*p.log = append(*p.log, p.name+":"+fmt.Sprintf(format, args...))
	}
}

func (p *hookPlugin) Name() string { return p.name }

func (p *hookPlugin) Init(_ context.Context, h Host) error {
	p.host = h
	p.record("init")
	return p.initErr
}

func (p *hookPlugin) SystemPrompt(context.Context) (string, error) { return p.prompt, nil }

func (p *hookPlugin) BeforeModel(_ context.Context, req *llm.Request) error {
	p.record("before_model")
	if p.beforeModel != nil {
		return p.beforeModel(req)
	}
	return nil
}

func (p *hookPlugin) AfterModel(_ context.Context, resp *llm.Response) error {
	p.record("after_model")
	if p.afterModel != nil {
		return p.afterModel(resp)
	}
	return nil
}

func (p *hookPlugin) BeforeTool(_ context.Context, c *llm.ToolCall) (Decision, error) {
	p.record("before_tool %s %s", c.Name, c.Arguments)
	if p.beforeTool != nil {
		return p.beforeTool(c)
	}
	return Allow(), nil
}

func (p *hookPlugin) AfterTool(_ context.Context, c llm.ToolCall, r *llm.ToolResult) error {
	p.record("after_tool %s error=%v", c.Name, r.IsError)
	if p.afterTool != nil {
		return p.afterTool(c, r)
	}
	return nil
}

func (p *hookPlugin) OnFinish(ctx context.Context, r *Result) error {
	p.record("finish %s", r.FinishReason)
	if p.onFinish != nil {
		return p.onFinish(ctx, r)
	}
	return nil
}

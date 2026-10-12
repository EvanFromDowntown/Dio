package session

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/EvanFromDowntown/Dio/agent"
	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/tool"
)

type step func(ctx context.Context, req *llm.Request) (*llm.Response, error)

type fakeClient struct {
	mu       sync.Mutex
	steps    []step
	requests [][]llm.Message
}

func (f *fakeClient) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	f.mu.Lock()
	f.requests = append(f.requests, slices.Clone(req.Messages))
	if len(f.steps) == 0 {
		f.mu.Unlock()
		return nil, errors.New("fake: unexpected call")
	}
	s := f.steps[0]
	f.steps = f.steps[1:]
	f.mu.Unlock()
	return s(ctx, req)
}

func (f *fakeClient) Stream(ctx context.Context, req *llm.Request) (llm.Stream, error) {
	resp, err := f.Generate(ctx, req)
	if err != nil {
		return nil, err
	}
	return &oneShot{resp: resp}, nil
}

type oneShot struct {
	resp *llm.Response
	done bool
}

func (s *oneShot) Next() bool {
	if s.done {
		return false
	}
	s.done = true
	return true
}
func (s *oneShot) Event() llm.StreamEvent {
	return llm.StreamEvent{Type: llm.EventDone, Response: s.resp}
}
func (s *oneShot) Err() error   { return nil }
func (s *oneShot) Close() error { return nil }

func text(s string) step {
	return func(context.Context, *llm.Request) (*llm.Response, error) {
		return &llm.Response{
			Content: []llm.ContentBlock{
				{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: "fake", Text: "thinking about " + s}},
				llm.TextBlock(s),
			},
			StopReason: llm.StopEndTurn, Usage: llm.Usage{InputTokens: 10, OutputTokens: 1},
		}, nil
	}
}

func callTool(id, name string) step {
	return func(context.Context, *llm.Request) (*llm.Response, error) {
		return &llm.Response{
			Content:    []llm.ContentBlock{llm.ToolCallBlock(llm.ToolCall{ID: id, Name: name, Arguments: json.RawMessage(`{"k": 1}`)})},
			StopReason: llm.StopToolUse, Usage: llm.Usage{InputTokens: 10, OutputTokens: 2},
		}, nil
	}
}

func setup(t *testing.T, store Store, client *fakeClient, maxTurns int, tools ...tool.Tool) (Runner, string) {
	t.Helper()
	a, err := agent.New(context.Background(), agent.Config{
		Client: client, Model: "m", MaxTurns: maxTurns, Tools: tools,
		Plugins: []agent.Plugin{NewRecorder(store)},
	})
	if err != nil {
		t.Fatal(err)
	}
	sess, err := store.Create(context.Background(), Info{Provider: "fake", Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	return Runner{Agent: a, Store: store}, sess.ID
}

func echo() tool.Tool {
	return &tool.Func{ToolName: "echo", Fn: func(_ context.Context, args json.RawMessage) (string, error) {
		return "echo " + string(args), nil
	}}
}

// assertPersisted checks that the stored history equals the run's history.
func assertPersisted(t *testing.T, store Store, id string, res *agent.Result) *Session {
	t.Helper()
	sess, err := store.Get(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(sess.Messages, res.Messages) {
		t.Errorf("persisted history differs from run history\nstored %+v\nrun    %+v", sess.Messages, res.Messages)
	}
	if PendingToolCalls(sess.Messages) != nil {
		t.Errorf("persisted history has unanswered tool calls")
	}
	return sess
}

func TestRunnerContinuesAcrossRuns(t *testing.T) {
	ctx := context.Background()
	for name, store := range stores(t) {
		t.Run(name, func(t *testing.T) {
			client := &fakeClient{steps: []step{callTool("c1", "echo"), text("noted"), text("it was blue")}}
			r, id := setup(t, store, client, 5, echo())

			res, err := r.Run(ctx, id, llm.UserText("Remember: my favourite colour is blue."))
			if err != nil {
				t.Fatal(err)
			}
			first := assertPersisted(t, store, id, res)
			if len(first.Messages) != 4 || first.Title != "Remember: my favourite colour is blue." {
				t.Errorf("after run 1: title=%q messages=%d", first.Title, len(first.Messages))
			}

			res, err = r.Run(ctx, id, llm.UserText("What is my favourite colour?"))
			if err != nil {
				t.Fatal(err)
			}
			second := assertPersisted(t, store, id, res)
			if len(second.Messages) != 6 || res.Text() != "it was blue" {
				t.Errorf("after run 2: %d messages, text %q", len(second.Messages), res.Text())
			}
			lastReq := client.requests[len(client.requests)-1]
			if !reflect.DeepEqual(lastReq, second.Messages[:5]) {
				t.Errorf("run 2 did not send the full persisted history")
			}
			if want := (llm.Usage{InputTokens: 30, OutputTokens: 4}); second.Usage != want {
				t.Errorf("usage = %+v", second.Usage)
			}
			if second.Title != first.Title {
				t.Errorf("title changed to %q", second.Title)
			}
		})
	}
}

func TestRunnerResumesAfterInterruption(t *testing.T) {
	ctx := context.Background()
	cases := []struct {
		name   string
		steps  []step
		tools  func(cancel context.CancelFunc) []tool.Tool
		check  func(error) bool
		reason agent.FinishReason
		extra  int
	}{
		{
			name: "canceled during tool execution",
			steps: []step{func(context.Context, *llm.Request) (*llm.Response, error) {
				return &llm.Response{Content: []llm.ContentBlock{
					llm.ToolCallBlock(llm.ToolCall{ID: "c1", Name: "cancel"}),
					llm.ToolCallBlock(llm.ToolCall{ID: "c2", Name: "cancel"}),
				}}, nil
			}},
			tools: func(cancel context.CancelFunc) []tool.Tool {
				return []tool.Tool{&tool.Func{ToolName: "cancel", Fn: func(context.Context, json.RawMessage) (string, error) {
					cancel()
					return "partial", nil
				}}}
			},
			check:  func(err error) bool { return errors.Is(err, context.Canceled) },
			reason: agent.FinishCanceled,
		},
		{
			name:   "max turns",
			steps:  []step{callTool("c1", "echo"), callTool("c2", "echo")},
			tools:  func(context.CancelFunc) []tool.Tool { return []tool.Tool{echo()} },
			check:  func(err error) bool { return errors.Is(err, agent.ErrMaxTurns) },
			reason: agent.FinishMaxTurns,
		},
		{
			name: "model error",
			steps: []step{callTool("c1", "echo"), func(context.Context, *llm.Request) (*llm.Response, error) {
				return nil, &llm.APIError{Provider: "fake", StatusCode: 503}
			}},
			tools:  func(context.CancelFunc) []tool.Tool { return []tool.Tool{echo()} },
			check:  func(err error) bool { return llm.IsRetryable(err) },
			reason: agent.FinishError,
		},
	}
	for name, store := range stores(t) {
		for _, tc := range cases {
			t.Run(name+"/"+tc.name, func(t *testing.T) {
				runCtx, cancel := context.WithCancel(ctx)
				defer cancel()
				client := &fakeClient{steps: append(tc.steps, text("resumed"))}
				r, id := setup(t, store, client, 2, tc.tools(cancel)...)

				res, err := r.Run(runCtx, id, llm.UserText("start"))
				if !tc.check(err) || res.FinishReason != tc.reason {
					t.Fatalf("err=%v reason=%s", err, res.FinishReason)
				}
				interrupted := assertPersisted(t, store, id, res)

				res, err = r.Run(ctx, id)
				if err != nil {
					t.Fatalf("resume: %v", err)
				}
				if res.Text() != "resumed" {
					t.Errorf("resume text = %q", res.Text())
				}
				resumed := assertPersisted(t, store, id, res)
				if !reflect.DeepEqual(resumed.Messages[:len(interrupted.Messages)], interrupted.Messages) {
					t.Error("resume rewrote earlier history")
				}
				lastReq := client.requests[len(client.requests)-1]
				if !reflect.DeepEqual(lastReq, interrupted.Messages) {
					t.Errorf("resume request = %+v\nwant %+v", lastReq, interrupted.Messages)
				}
			})
		}
	}
}

func TestRunnerRepairsCrashedSession(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	client := &fakeClient{steps: []step{text("recovered")}}
	r, id := setup(t, store, client, 3)
	// A crash between recording a tool request and its results.
	store.AppendMessages(ctx, id,
		llm.UserText("do it"),
		llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.ToolCallBlock(llm.ToolCall{ID: "c9", Name: "slow"})}},
	)

	res, err := r.Run(ctx, id, llm.UserText("are you there?"))
	if err != nil {
		t.Fatal(err)
	}
	sess := assertPersisted(t, store, id, res)
	repaired := sess.Messages[2]
	if len(repaired.Content) != 1 || repaired.Content[0].ToolResult == nil ||
		repaired.Content[0].ToolResult.ToolCallID != "c9" || !repaired.Content[0].ToolResult.IsError {
		t.Errorf("repair message = %+v", repaired)
	}
	if sess.Messages[3].Text() != "are you there?" || res.Text() != "recovered" {
		t.Errorf("history = %+v", sess.Messages)
	}
}

func TestRunnerStreamAndErrors(t *testing.T) {
	ctx := context.Background()
	store := NewMemoryStore()
	client := &fakeClient{steps: []step{text("streamed")}}
	r, id := setup(t, store, client, 3)

	var final *agent.Result
	for ev, err := range r.Stream(ctx, id, llm.UserText("hi")) {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Type == agent.EventDone {
			final = ev.Result
		}
	}
	if final == nil || final.Text() != "streamed" {
		t.Fatalf("final = %+v", final)
	}
	assertPersisted(t, store, id, final)

	if _, err := r.Run(ctx, "missing", llm.UserText("x")); !errors.Is(err, ErrNotFound) {
		t.Errorf("missing session: %v", err)
	}
	for _, err := range r.Stream(ctx, "missing") {
		if !errors.Is(err, ErrNotFound) {
			t.Errorf("stream missing session: %v", err)
		}
	}
}

func TestRecorderWithoutSessionIsNoop(t *testing.T) {
	store := NewMemoryStore()
	a, err := agent.New(context.Background(), agent.Config{
		Client: &fakeClient{steps: []step{text("ok")}}, Model: "m",
		Plugins: []agent.Plugin{NewRecorder(store)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Run(context.Background(), []llm.Message{llm.UserText("x")}); err != nil {
		t.Fatal(err)
	}
	if list, _ := store.List(context.Background()); len(list) != 0 {
		t.Errorf("recorder wrote without a session: %+v", list)
	}
}

func TestRecorderStoreFailureAbortsRun(t *testing.T) {
	store := NewMemoryStore()
	client := &fakeClient{steps: []step{callTool("c1", "echo"), text("unreachable")}}
	r, id := setup(t, store, client, 3, echo())
	// Input and the assistant response persist; the tool results fail.
	failing := &failingStore{Store: store, failAfter: 2}
	r.Agent, _ = agent.New(context.Background(), agent.Config{
		Client: client, Model: "m", Tools: []tool.Tool{echo()},
		Plugins: []agent.Plugin{NewRecorder(failing)},
	})
	r.Store = failing
	_, err := r.Run(context.Background(), id, llm.UserText("x"))
	if err == nil || !strings.Contains(err.Error(), "disk full") {
		t.Errorf("err = %v", err)
	}
}

type failingStore struct {
	Store
	mu        sync.Mutex
	appends   int
	failAfter int
}

func (f *failingStore) AppendMessages(ctx context.Context, id string, msgs ...llm.Message) error {
	f.mu.Lock()
	f.appends++
	n := f.appends
	f.mu.Unlock()
	if n > f.failAfter {
		return errors.New("disk full")
	}
	return f.Store.AppendMessages(ctx, id, msgs...)
}

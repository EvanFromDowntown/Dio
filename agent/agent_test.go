package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/tool"
)

func (p *hookPlugin) Tools() []tool.Tool {
	var out []tool.Tool
	for _, d := range p.tools {
		out = append(out, &tool.Func{ToolName: d.name, ToolDescription: d.name + " tool", Fn: d.fn})
	}
	return out
}

func echoTool() tool.Tool {
	return &tool.Func{
		ToolName: "echo", ToolDescription: "Echo the text argument",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}}}`),
		Fn: func(_ context.Context, args json.RawMessage) (string, error) {
			var a struct{ Text string }
			if err := json.Unmarshal(args, &a); err != nil {
				return "", err
			}
			return "echo: " + a.Text, nil
		},
	}
}

func newAgent(t *testing.T, client llm.Client, cfg Config) *Agent {
	t.Helper()
	cfg.Client = client
	if cfg.Model == "" {
		cfg.Model = "test-model"
	}
	a, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return a
}

func toolResultsOf(m llm.Message) []llm.ToolResult {
	var out []llm.ToolResult
	for _, b := range m.Content {
		if b.ToolResult != nil {
			out = append(out, *b.ToolResult)
		}
	}
	return out
}

func TestMultiTurnToolCalls(t *testing.T) {
	var log []string
	adder := toolDef{name: "add", fn: func(_ context.Context, args json.RawMessage) (string, error) {
		var a struct{ A, B int }
		json.Unmarshal(args, &a)
		return fmt.Sprint(a.A + a.B), nil
	}}
	p := &hookPlugin{name: "math", log: &log, prompt: "  You can add numbers.  ", tools: []toolDef{adder}}
	silent := &hookPlugin{name: "silent"}

	client := newFake(
		reply(toolResp(call("c1", "echo", `{"text":"hi"}`), call("c2", "add", `{"a":2,"b":3}`))),
		reply(toolResp(call("c3", "echo", `{"text":"again"}`))),
		reply(textResp("all done")),
	)
	a := newAgent(t, client, Config{
		System: "Base prompt.", Tools: []tool.Tool{echoTool()}, Plugins: []Plugin{p, silent},
		MaxTokens: 100, Temperature: llm.Ptr(0.5),
	})
	input := []llm.Message{llm.UserText("go")}
	res, err := a.Run(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}

	if res.FinishReason != FinishCompleted || res.Err != nil || res.Turns != 3 || res.Text() != "all done" {
		t.Errorf("result: reason=%s turns=%d text=%q err=%v", res.FinishReason, res.Turns, res.Text(), res.Err)
	}
	if want := (llm.Usage{InputTokens: 30, OutputTokens: 5}); res.Usage != want {
		t.Errorf("usage = %+v", res.Usage)
	}
	if len(input) != 1 {
		t.Error("input slice was modified")
	}

	// user, assistant(2 calls), results, assistant(1 call), results, assistant(text)
	if len(res.Messages) != 6 {
		t.Fatalf("history has %d messages", len(res.Messages))
	}
	wantResults := []llm.ToolResult{{ToolCallID: "c1", Content: "echo: hi"}, {ToolCallID: "c2", Content: "5"}}
	if got := toolResultsOf(res.Messages[2]); !reflect.DeepEqual(got, wantResults) || res.Messages[2].Role != llm.RoleUser {
		t.Errorf("turn 1 results = %+v", got)
	}
	if got := toolResultsOf(res.Messages[4]); !reflect.DeepEqual(got, []llm.ToolResult{{ToolCallID: "c3", Content: "echo: again"}}) {
		t.Errorf("turn 2 results = %+v", got)
	}

	if len(client.requests) != 3 {
		t.Fatalf("model called %d times", len(client.requests))
	}
	first := client.requests[0]
	if first.System != "Base prompt.\n\nYou can add numbers." {
		t.Errorf("system = %q", first.System)
	}
	if first.Model != "test-model" || first.MaxTokens != 100 || *first.Temperature != 0.5 {
		t.Errorf("request params = %+v", first)
	}
	var names []string
	for _, tl := range first.Tools {
		names = append(names, tl.Name)
	}
	if !reflect.DeepEqual(names, []string{"echo", "add"}) {
		t.Errorf("tools = %v", names)
	}
	for i, n := range []int{1, 3, 5} {
		if got := len(client.requests[i].Messages); got != n {
			t.Errorf("request %d has %d messages, want %d", i, got, n)
		}
	}

	wantLog := []string{
		"math:init",
		"math:before_model", "math:after_model",
		`math:before_tool echo {"text":"hi"}`, "math:after_tool echo error=false",
		`math:before_tool add {"a":2,"b":3}`, "math:after_tool add error=false",
		"math:before_model", "math:after_model",
		`math:before_tool echo {"text":"again"}`, "math:after_tool echo error=false",
		"math:before_model", "math:after_model",
		"math:finish completed",
	}
	if !reflect.DeepEqual(log, wantLog) {
		t.Errorf("hook log:\n%s", strings.Join(log, "\n"))
	}
}

func TestToolErrorsAreReturnedToModel(t *testing.T) {
	failing := toolDef{name: "fail", fn: func(context.Context, json.RawMessage) (string, error) {
		return "", errors.New("disk on fire")
	}}
	panicking := toolDef{name: "boom", fn: func(context.Context, json.RawMessage) (string, error) {
		panic("kaboom")
	}}
	client := newFake(
		reply(toolResp(call("c1", "fail", `{}`), call("c2", "missing", `{}`), call("c3", "boom", `{}`), call("c4", "echo", `not json`))),
		reply(textResp("recovered")),
	)
	a := newAgent(t, client, Config{Tools: []tool.Tool{echoTool()}, Plugins: []Plugin{&hookPlugin{name: "p", tools: []toolDef{failing, panicking}}}})
	res, err := a.Run(context.Background(), []llm.Message{llm.UserText("go")})
	if err != nil {
		t.Fatal(err)
	}
	got := toolResultsOf(res.Messages[2])
	if len(got) != 4 {
		t.Fatalf("results = %+v", got)
	}
	checks := []string{"disk on fire", `Unknown tool "missing"`, `Tool "boom" panicked: kaboom`, "invalid character"}
	for i, want := range checks {
		if !got[i].IsError || !strings.Contains(got[i].Content, want) {
			t.Errorf("result %d = %+v, want error containing %q", i, got[i], want)
		}
	}
	if res.Text() != "recovered" || client.calls() != 2 {
		t.Errorf("loop did not continue: text=%q calls=%d", res.Text(), client.calls())
	}
}

func TestToolHookDeny(t *testing.T) {
	var log []string
	executed := false
	guarded := toolDef{name: "rm", fn: func(context.Context, json.RawMessage) (string, error) {
		executed = true
		return "deleted", nil
	}}
	guard := &hookPlugin{name: "guard", log: &log, tools: []toolDef{guarded}, beforeTool: func(c *llm.ToolCall) (Decision, error) {
		if c.Name == "rm" {
			return Deny("destructive commands are not allowed"), nil
		}
		return Allow(), nil
	}}
	audit := &hookPlugin{name: "audit", log: &log}
	client := newFake(reply(toolResp(call("c1", "rm", `{"path":"/"}`))), reply(textResp("ok")))
	a := newAgent(t, client, Config{Plugins: []Plugin{guard, audit}})

	res, err := a.Run(context.Background(), []llm.Message{llm.UserText("go")})
	if err != nil {
		t.Fatal(err)
	}
	if executed {
		t.Error("denied tool was executed")
	}
	want := llm.ToolResult{ToolCallID: "c1", Content: "Tool call denied: destructive commands are not allowed", IsError: true}
	if got := toolResultsOf(res.Messages[2]); !reflect.DeepEqual(got, []llm.ToolResult{want}) {
		t.Errorf("results = %+v", got)
	}
	// The second BeforeTool hook is skipped; every AfterTool hook still runs.
	var toolLog []string
	for _, l := range log {
		if strings.Contains(l, "_tool") {
			toolLog = append(toolLog, l)
		}
	}
	wantLog := []string{`guard:before_tool rm {"path":"/"}`, "guard:after_tool rm error=true", "audit:after_tool rm error=true"}
	if !reflect.DeepEqual(toolLog, wantLog) {
		t.Errorf("tool hook log = %v", toolLog)
	}
}

func TestToolHookModifyArguments(t *testing.T) {
	var log []string
	var gotArgs string
	writer := toolDef{name: "write", fn: func(_ context.Context, args json.RawMessage) (string, error) {
		gotArgs = string(args)
		return "written", nil
	}}
	sandbox := &hookPlugin{name: "sandbox", log: &log, tools: []toolDef{writer}, beforeTool: func(c *llm.ToolCall) (Decision, error) {
		c.ID = "tampered"
		return ModifyArguments(json.RawMessage(`{"path":"/sandbox/a.txt"}`)), nil
	}}
	audit := &hookPlugin{name: "audit", log: &log, afterTool: func(_ llm.ToolCall, r *llm.ToolResult) error {
		r.Content += " (audited)"
		return nil
	}}
	client := newFake(reply(toolResp(call("c1", "write", `{"path":"/etc/passwd"}`))), reply(textResp("ok")))
	a := newAgent(t, client, Config{Plugins: []Plugin{sandbox, audit}})

	res, err := a.Run(context.Background(), []llm.Message{llm.UserText("go")})
	if err != nil {
		t.Fatal(err)
	}
	if gotArgs != `{"path":"/sandbox/a.txt"}` {
		t.Errorf("tool received %s", gotArgs)
	}
	if !reflect.DeepEqual(log[6:8], []string{`sandbox:before_tool write {"path":"/etc/passwd"}`, `audit:before_tool write {"path":"/sandbox/a.txt"}`}) {
		t.Errorf("hooks did not see arguments in order: %v", log)
	}
	if got := toolResultsOf(res.Messages[2]); !reflect.DeepEqual(got, []llm.ToolResult{{ToolCallID: "c1", Content: "written (audited)"}}) {
		t.Errorf("results = %+v", got)
	}
	if args := string(res.Messages[1].ToolCalls()[0].Arguments); args != `{"path":"/etc/passwd"}` {
		t.Errorf("history call arguments rewritten to %s", args)
	}
}

func TestModelHooks(t *testing.T) {
	p := &hookPlugin{
		name: "p",
		beforeModel: func(r *llm.Request) error {
			r.Temperature = llm.Ptr(0.0)
			r.Messages = append(r.Messages, llm.UserText("injected"))
			return nil
		},
		afterModel: func(r *llm.Response) error {
			r.Content[0].Text = strings.ToUpper(r.Content[0].Text)
			return nil
		},
	}
	client := newFake(reply(textResp("quiet")))
	a := newAgent(t, client, Config{Plugins: []Plugin{p}})
	res, err := a.Run(context.Background(), []llm.Message{llm.UserText("go")})
	if err != nil {
		t.Fatal(err)
	}
	req := client.requests[0]
	if *req.Temperature != 0 || len(req.Messages) != 2 {
		t.Errorf("BeforeModel changes not applied: %+v", req)
	}
	if len(res.Messages) != 2 || res.Messages[1].Text() != "QUIET" {
		t.Errorf("history = %+v", res.Messages)
	}
}

func TestHookErrorsAbort(t *testing.T) {
	boom := errors.New("policy engine down")
	tests := []struct {
		name   string
		plugin *hookPlugin
	}{
		{"before model", &hookPlugin{name: "p", beforeModel: func(*llm.Request) error { return boom }}},
		{"after model", &hookPlugin{name: "p", afterModel: func(*llm.Response) error { return boom }}},
		{"before tool", &hookPlugin{name: "p", beforeTool: func(*llm.ToolCall) (Decision, error) { return Decision{}, boom }}},
		{"after tool", &hookPlugin{name: "p", afterTool: func(llm.ToolCall, *llm.ToolResult) error { return boom }}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := newFake(reply(toolResp(call("c1", "echo", `{}`), call("c2", "echo", `{}`))), reply(textResp("unreachable")))
			a := newAgent(t, client, Config{Tools: []tool.Tool{echoTool()}, Plugins: []Plugin{tt.plugin}})
			res, err := a.Run(context.Background(), []llm.Message{llm.UserText("go")})
			if !errors.Is(err, boom) || res.FinishReason != FinishError || !errors.Is(res.Err, boom) {
				t.Fatalf("err=%v reason=%s", err, res.FinishReason)
			}
			last := res.Messages[len(res.Messages)-1]
			if results := toolResultsOf(last); len(results) > 0 {
				if len(results) != 2 || !results[1].IsError || !strings.Contains(results[1].Content, "interrupted") {
					t.Errorf("tool results not backfilled: %+v", results)
				}
			}
		})
	}
}

func TestMaxTurns(t *testing.T) {
	var executed int
	counter := toolDef{name: "tick", fn: func(context.Context, json.RawMessage) (string, error) {
		executed++
		return "tock", nil
	}}
	loop := func(context.Context, *llm.Request) (*llm.Response, error) {
		return toolResp(call("c", "tick", `{}`)), nil
	}
	client := newFake(loop, loop, loop)
	var finished *Result
	p := &hookPlugin{name: "p", tools: []toolDef{counter}, onFinish: func(_ context.Context, r *Result) error {
		finished = r
		return nil
	}}
	a := newAgent(t, client, Config{MaxTurns: 2, Plugins: []Plugin{p}})
	res, err := a.Run(context.Background(), []llm.Message{llm.UserText("go")})
	if !errors.Is(err, ErrMaxTurns) {
		t.Fatalf("err = %v", err)
	}
	if res.FinishReason != FinishMaxTurns || res.Turns != 2 || client.calls() != 2 || executed != 2 {
		t.Errorf("reason=%s turns=%d calls=%d executed=%d", res.FinishReason, res.Turns, client.calls(), executed)
	}
	if last := res.Messages[len(res.Messages)-1]; len(toolResultsOf(last)) != 1 {
		t.Errorf("history should end with tool results, got %+v", last)
	}
	if finished != res {
		t.Error("OnFinish did not receive the result")
	}
}

func TestDefaultMaxTurns(t *testing.T) {
	a := newAgent(t, newFake(), Config{})
	if a.cfg.MaxTurns != DefaultMaxTurns {
		t.Errorf("MaxTurns = %d", a.cfg.MaxTurns)
	}
}

func TestCancellation(t *testing.T) {
	t.Run("before first turn", func(t *testing.T) {
		client := newFake(reply(textResp("x")))
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		res, err := newAgent(t, client, Config{}).Run(ctx, []llm.Message{llm.UserText("go")})
		if !errors.Is(err, context.Canceled) || res.FinishReason != FinishCanceled || client.calls() != 0 {
			t.Errorf("err=%v reason=%s calls=%d", err, res.FinishReason, client.calls())
		}
	})

	t.Run("during tool execution", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		var secondRan bool
		slow := toolDef{name: "slow", fn: func(ctx context.Context, _ json.RawMessage) (string, error) {
			cancel()
			<-ctx.Done()
			return "", ctx.Err()
		}}
		second := toolDef{name: "second", fn: func(context.Context, json.RawMessage) (string, error) {
			secondRan = true
			return "", nil
		}}
		var finishCtxErr error = errors.New("not called")
		p := &hookPlugin{name: "p", tools: []toolDef{slow, second}, onFinish: func(ctx context.Context, _ *Result) error {
			finishCtxErr = ctx.Err()
			return nil
		}}
		client := newFake(reply(toolResp(call("c1", "slow", `{}`), call("c2", "second", `{}`))), reply(textResp("unreachable")))
		res, err := newAgent(t, client, Config{Plugins: []Plugin{p}}).Run(ctx, []llm.Message{llm.UserText("go")})
		if !errors.Is(err, context.Canceled) || res.FinishReason != FinishCanceled {
			t.Fatalf("err=%v reason=%s", err, res.FinishReason)
		}
		if secondRan || client.calls() != 1 {
			t.Errorf("run continued after cancel: secondRan=%v calls=%d", secondRan, client.calls())
		}
		results := toolResultsOf(res.Messages[len(res.Messages)-1])
		if len(results) != 2 || !results[0].IsError || !strings.Contains(results[1].Content, "interrupted") {
			t.Errorf("results = %+v", results)
		}
		if finishCtxErr != nil {
			t.Errorf("OnFinish context error = %v", finishCtxErr)
		}
	})

	t.Run("model call canceled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		client := newFake(func(ctx context.Context, _ *llm.Request) (*llm.Response, error) {
			cancel()
			return nil, ctx.Err()
		})
		res, err := newAgent(t, client, Config{}).Run(ctx, []llm.Message{llm.UserText("go")})
		if !errors.Is(err, context.Canceled) || res.FinishReason != FinishCanceled {
			t.Errorf("err=%v reason=%s", err, res.FinishReason)
		}
	})
}

func TestModelErrorAborts(t *testing.T) {
	apiErr := &llm.APIError{Provider: "fake", StatusCode: 429, Message: "slow down"}
	client := newFake(func(context.Context, *llm.Request) (*llm.Response, error) { return nil, apiErr })
	res, err := newAgent(t, client, Config{}).Run(context.Background(), []llm.Message{llm.UserText("go")})
	var got *llm.APIError
	if !errors.As(err, &got) || !llm.IsRetryable(err) || res.FinishReason != FinishError || len(res.Messages) != 1 {
		t.Errorf("err=%v reason=%s", err, res.FinishReason)
	}
}

func TestOnFinishErrorIsReported(t *testing.T) {
	p := &hookPlugin{name: "p", onFinish: func(context.Context, *Result) error { return errors.New("flush failed") }}
	res, err := newAgent(t, newFake(reply(textResp("hi"))), Config{Plugins: []Plugin{p}}).Run(context.Background(), []llm.Message{llm.UserText("go")})
	if err == nil || !strings.Contains(err.Error(), "flush failed") || res.FinishReason != FinishCompleted {
		t.Errorf("err=%v reason=%s", err, res.FinishReason)
	}
}

func TestStream(t *testing.T) {
	client := newFake(
		reply(toolResp(call("c1", "echo", `{"text":"x"}`))),
		reply(textResp("done")),
	)
	a := newAgent(t, client, Config{Tools: []tool.Tool{echoTool()}})

	var types []string
	var text strings.Builder
	var final *Result
	for ev, err := range a.Stream(context.Background(), []llm.Message{llm.UserText("go")}) {
		if err != nil {
			t.Fatal(err)
		}
		name := string(ev.Type)
		if ev.Type == EventModel {
			name += "/" + string(ev.LLM.Type)
			if ev.LLM.Type == llm.EventTextDelta {
				text.WriteString(ev.LLM.Delta)
			}
		}
		types = append(types, fmt.Sprintf("%d:%s", ev.Turn, name))
		if ev.Type == EventDone {
			final = ev.Result
		}
	}
	want := []string{
		"1:turn_start",
		"1:model/start", "1:model/content_start", "1:model/tool_call_delta", "1:model/content_stop", "1:model/done",
		"1:model_response", "1:tool_call", "1:tool_result",
		"2:turn_start",
		"2:model/start", "2:model/content_start", "2:model/text_delta", "2:model/content_stop", "2:model/done",
		"2:model_response",
		"2:done",
	}
	if !reflect.DeepEqual(types, want) {
		t.Errorf("events:\n%s", strings.Join(types, "\n"))
	}
	if text.String() != "done" || final == nil || final.FinishReason != FinishCompleted || len(final.Messages) != 4 {
		t.Errorf("final = %+v", final)
	}
	for i, s := range client.streams {
		if !s.closed {
			t.Errorf("model stream %d not closed", i)
		}
	}
}

func TestStreamErrorEndsWithDone(t *testing.T) {
	client := newFake(func(context.Context, *llm.Request) (*llm.Response, error) { return nil, errors.New("upstream down") })
	var last Event
	var lastErr error
	for ev, err := range newAgent(t, client, Config{}).Stream(context.Background(), []llm.Message{llm.UserText("go")}) {
		last, lastErr = ev, err
	}
	if last.Type != EventDone || lastErr == nil || last.Result.FinishReason != FinishError {
		t.Errorf("last event = %+v, err = %v", last, lastErr)
	}
}

func TestStreamConsumerStops(t *testing.T) {
	var finished *Result
	p := &hookPlugin{name: "p", onFinish: func(_ context.Context, r *Result) error { finished = r; return nil }}
	client := newFake(reply(textResp("hello")))
	a := newAgent(t, client, Config{Plugins: []Plugin{p}})
	for ev := range a.Stream(context.Background(), []llm.Message{llm.UserText("go")}) {
		if ev.Type == EventModel && ev.LLM.Type == llm.EventTextDelta {
			break
		}
	}
	if finished == nil || finished.FinishReason != FinishCanceled || !errors.Is(finished.Err, ErrStopped) {
		t.Errorf("finished = %+v", finished)
	}
	if !client.streams[0].closed {
		t.Error("model stream not closed after consumer stopped")
	}
}

func TestNew(t *testing.T) {
	ctx := context.Background()
	client := newFake()
	cases := []struct {
		name string
		cfg  Config
		want string
	}{
		{"no client", Config{Model: "m"}, "Client"},
		{"no model", Config{Client: client}, "Model"},
		{"nil plugin", Config{Client: client, Model: "m", Plugins: []Plugin{nil}}, "nil plugin"},
		{"duplicate plugin", Config{Client: client, Model: "m", Plugins: []Plugin{&hookPlugin{name: "a"}, &hookPlugin{name: "a"}}}, "duplicate plugin"},
		{"init error", Config{Client: client, Model: "m", Plugins: []Plugin{&hookPlugin{name: "a", initErr: errors.New("no creds")}}}, "no creds"},
		{"duplicate tool", Config{Client: client, Model: "m", Tools: []tool.Tool{echoTool()},
			Plugins: []Plugin{&hookPlugin{name: "a", tools: []toolDef{{name: "echo"}}}}}, `duplicate tool name "echo"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := New(ctx, tc.cfg); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("err = %v, want %q", err, tc.want)
			}
		})
	}

	var log []string
	p1, p2 := &hookPlugin{name: "one", log: &log}, &hookPlugin{name: "two", log: &log}
	a, err := New(ctx, Config{Client: client, Model: "m", Plugins: []Plugin{p1, p2}})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(log, []string{"one:init", "two:init"}) || p1.host == nil || p1.host.Logger() == nil {
		t.Errorf("init log = %v", log)
	}
	if len(a.Tools()) != 0 {
		t.Errorf("tools = %v", a.Tools())
	}
}

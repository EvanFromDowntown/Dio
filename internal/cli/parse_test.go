package cli

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/EvanFromDowntown/Dio/agent"
	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/session"
)

func TestParse(t *testing.T) {
	env := map[string]string{"DIO_PROVIDER": "anthropic", "DIO_MODEL": "env-model", "DIO_BASE_URL": "https://env/v1"}
	getenv := func(k string) string { return env[k] }

	o, err := Parse(nil, noEnv, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if o.Provider != DefaultProvider || o.APIKeyEnv != "DIO_API_KEY" || o.Root != "." || o.MaxTurns != 20 ||
		o.MaxTokens != 4096 || o.Temperature != nil || o.Explicit("provider") || o.Command != "" {
		t.Errorf("defaults = %+v", o)
	}

	o, err = Parse([]string{"-model", "flag-model", "-temperature", "0.3", "-reasoning-effort", "low", "hello", "world"}, getenv, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if o.Provider != "anthropic/messages" || o.Model != "flag-model" || o.BaseURL != "https://env/v1" ||
		*o.Temperature != 0.3 || o.ReasoningEffort != "low" || o.Prompt != "hello world" ||
		!o.Explicit("provider") || !o.Explicit("model") {
		t.Errorf("env and flags = %+v", o)
	}

	for legacy, want := range map[string]string{"chatcompletions": "openai/chat-completions", "openai": "openai/responses", "anthropic": "anthropic/messages", "openai/responses": "openai/responses"} {
		if o, err := Parse([]string{"-provider", legacy}, noEnv, io.Discard); err != nil || o.Provider != want {
			t.Errorf("-provider %s = %v, %v", legacy, o, err)
		}
	}

	o, err = Parse([]string{"sessions", "-sessions-dir", "/tmp/x"}, getenv, io.Discard)
	if err != nil || o.Command != "sessions" || o.SessionsDir != "/tmp/x" {
		t.Errorf("sessions subcommand = %+v, %v", o, err)
	}

	for _, bad := range [][]string{
		{"-provider", "gemini"},
		{"-temperature", "3"},
		{"-temperature", "hot"},
		{"-reasoning-effort", "max"},
		{"-max-turns", "0"},
		{"sessions", "extra"},
		{"-unknown"},
	} {
		if _, err := Parse(bad, noEnv, io.Discard); err == nil {
			t.Errorf("Parse(%v) should fail", bad)
		}
	}
	var usage bytes.Buffer
	if _, err := Parse([]string{"-h"}, noEnv, &usage); !errors.Is(err, ErrHelp) || !strings.Contains(usage.String(), "dio sessions") {
		t.Errorf("help: %v\n%s", err, usage.String())
	}
}

func TestParseCommand(t *testing.T) {
	tests := map[string]Command{
		"":              {Kind: CmdEmpty},
		"   ":           {Kind: CmdEmpty},
		"hello":         {Kind: CmdMessage, Text: "hello"},
		"  hi there ":   {Kind: CmdMessage, Text: "hi there"},
		"/quit":         {Kind: CmdQuit},
		"/exit":         {Kind: CmdQuit},
		"/EXIT":         {Kind: CmdQuit},
		"/new":          {Kind: CmdNew},
		"/sessions":     {Kind: CmdSessions},
		"/help":         {Kind: CmdHelp},
		"/frobnicate x": {Kind: CmdUnknown, Text: "/frobnicate"},
		"//etc/hosts?":  {Kind: CmdMessage, Text: "/etc/hosts?"},
	}
	for in, want := range tests {
		if got := ParseCommand(in); got != want {
			t.Errorf("ParseCommand(%q) = %+v, want %+v", in, got, want)
		}
	}
}

func renderEvents(failTool bool) []agent.Event {
	call := llm.ToolCall{ID: "c1", Name: "read_file", Arguments: json.RawMessage(`{"path": "notes.txt", "limit": 10}`)}
	result := llm.ToolResult{ToolCallID: "c1", Content: "\nline one\nline two"}
	if failTool {
		result = llm.ToolResult{ToolCallID: "c1", Content: "open notes.txt: no such file", IsError: true}
	}
	return []agent.Event{
		{Type: agent.EventModel, LLM: &llm.StreamEvent{Type: llm.EventStart}},
		{Type: agent.EventModel, LLM: &llm.StreamEvent{Type: llm.EventReasoningDelta, Index: 0, Delta: "hmm"}},
		{Type: agent.EventModel, LLM: &llm.StreamEvent{Type: llm.EventContentStop, Index: 0, Block: &llm.ContentBlock{Type: llm.ContentReasoning}}},
		{Type: agent.EventModel, LLM: &llm.StreamEvent{Type: llm.EventTextDelta, Index: 1, Delta: "我来"}},
		{Type: agent.EventModel, LLM: &llm.StreamEvent{Type: llm.EventTextDelta, Index: 1, Delta: "读取。"}},
		{Type: agent.EventModel, LLM: &llm.StreamEvent{Type: llm.EventToolCallDelta, Index: 2, Delta: `{"path"`}},
		{Type: agent.EventModelResponse, Response: &llm.Response{}},
		{Type: agent.EventToolCall, ToolCall: &call},
		{Type: agent.EventToolResult, ToolCall: &call, ToolResult: &result},
		{Type: agent.EventModel, LLM: &llm.StreamEvent{Type: llm.EventTextDelta, Delta: "完成"}},
	}
}

func TestRendererSharedWriter(t *testing.T) {
	var out bytes.Buffer
	r := NewRenderer(&out, &out, false)
	for _, ev := range renderEvents(false) {
		r.Event(ev)
	}
	r.Finish()
	want := "我来读取。\n[工具] read_file(limit=10, path=notes.txt)\n[结果] read_file：18 字节 · line one\n完成\n"
	if out.String() != want {
		t.Errorf("got\n%q\nwant\n%q", out.String(), want)
	}
}

func TestRendererSplitWritersWithReasoning(t *testing.T) {
	var answer, info bytes.Buffer
	r := NewRenderer(&answer, &info, true)
	for _, ev := range renderEvents(true) {
		r.Event(ev)
	}
	r.Finish()
	if answer.String() != "我来读取。\n完成\n" {
		t.Errorf("answer = %q", answer.String())
	}
	want := "[思考] hmm\n[工具] read_file(limit=10, path=notes.txt)\n[出错] read_file：open notes.txt: no such file\n"
	if info.String() != want {
		t.Errorf("info = %q, want %q", info.String(), want)
	}
}

func TestSummarizeArgs(t *testing.T) {
	tests := []struct{ in, want string }{
		{`{"path":"a.txt"}`, "path=a.txt"},
		{`{"b":[1,2],"a":{"x":true}}`, `a={"x":true}, b=[1,2]`},
		{`{}`, ""},
		{`not   json`, "not json"},
		{`{"text":"` + strings.Repeat("长", 50) + `"}`, "text=" + strings.Repeat("长", 40) + "…"},
	}
	for _, tt := range tests {
		if got := SummarizeArgs(json.RawMessage(tt.in), 100); got != tt.want {
			t.Errorf("SummarizeArgs(%s) = %q, want %q", tt.in, got, tt.want)
		}
	}
	long := `{"a":"` + strings.Repeat("x", 30) + `","b":"` + strings.Repeat("y", 30) + `"}`
	if got := SummarizeArgs(json.RawMessage(long), 20); len([]rune(got)) != 21 {
		t.Errorf("total truncation = %q", got)
	}
}

func TestFormatUsageAndSessions(t *testing.T) {
	res := &agent.Result{Turns: 2, Usage: llm.Usage{InputTokens: 10, OutputTokens: 3}}
	if got := FormatUsage(res); got != "（2 次模型调用 · 输入 10 / 输出 3 tokens）" {
		t.Errorf("usage = %q", got)
	}

	var buf bytes.Buffer
	WriteSessions(&buf, nil, "")
	if !strings.Contains(buf.String(), "还没有会话") {
		t.Errorf("empty list = %q", buf.String())
	}
	buf.Reset()
	WriteSessions(&buf, []session.Info{
		{ID: "b", Provider: "openai/responses", Model: "m", MessageCount: 4, Usage: llm.Usage{InputTokens: 10, OutputTokens: 2}, Title: "第二个"},
		{ID: "a", Model: "m", Title: strings.Repeat("长", 50)},
	}, "b")
	out := buf.String()
	for _, want := range []string{"* b", "10/2", "openai/responses · m", strings.Repeat("长", 40) + "…"} {
		if !strings.Contains(out, want) {
			t.Errorf("table missing %q:\n%s", want, out)
		}
	}
}

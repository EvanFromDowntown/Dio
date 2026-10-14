package chatcompletions

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/EvanFromDowntown/Dio/internal/testutil"
	"github.com/EvanFromDowntown/Dio/llm"
)

// Fixtures under testdata are sanitized captures from an OpenAI-compatible
// gateway serving a DeepSeek model.
func fixture(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func newClient(srv *httptest.Server, opts ...Option) *Client {
	return New(append([]Option{WithBaseURL(srv.URL + "/standard/v1/"), WithAPIKey("sk-test"), WithHTTPClient(srv.Client())}, opts...)...)
}

func simpleRequest() *llm.Request {
	return &llm.Request{Model: "deepseek-flash", Messages: []llm.Message{llm.UserText("x")}}
}

const okResponse = `{"id":"c","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}]}`

func TestRequestSerialization(t *testing.T) {
	srv, got := testutil.Server(t, 200, "application/json", okResponse)
	c := newClient(srv, WithHeader("X-Gateway", "dio"))
	req := &llm.Request{
		Model:  "deepseek-flash",
		System: "You are helpful.",
		Messages: []llm.Message{
			llm.UserText("Weather in Beijing?"),
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				reasoningBlock("Need the tool."),
				{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: "anthropic", Text: "foreign"}},
				llm.ToolCallBlock(llm.ToolCall{ID: "call_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"Beijing"}`)}),
				llm.ToolCallBlock(llm.ToolCall{ID: "call_2", Name: "now"}),
			}},
			{Role: llm.RoleUser, Content: []llm.ContentBlock{
				llm.ToolResultBlock(llm.ToolResult{ToolCallID: "call_1", Content: "Sunny, 25C"}),
				llm.ToolResultBlock(llm.ToolResult{ToolCallID: "call_2", Content: "clock broken", IsError: true}),
				llm.TextBlock("Also, "),
				llm.TextBlock("what about this?"),
				llm.ImageURLBlock("https://x/img.png"),
				llm.ImageDataBlock("image/png", "AAAA"),
			}},
			llm.AssistantText("Looks like a cat."),
		},
		Tools: []llm.Tool{
			{Name: "get_weather", Description: "Get weather", InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)},
			{Name: "now"},
		},
		ToolChoice:    &llm.ToolChoice{Mode: llm.ToolChoiceAuto},
		MaxTokens:     1024,
		Temperature:   llm.Ptr(1.0),
		TopP:          llm.Ptr(0.95),
		StopSequences: []string{"\n\n"},
		Reasoning:     &llm.ReasoningConfig{Effort: "low", BudgetTokens: 999},
		Extra:         map[string]any{"response_format": map[string]string{"type": "json_object"}},
	}
	if _, err := c.Generate(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if got.Method != http.MethodPost || got.Path != "/standard/v1/chat/completions" {
		t.Errorf("request line: %s %s", got.Method, got.Path)
	}
	if got.Header.Get("Authorization") != "Bearer sk-test" || got.Header.Get("X-Gateway") != "dio" {
		t.Errorf("headers = %v", got.Header)
	}
	testutil.AssertJSON(t, got.Body, `{
		"model": "deepseek-flash",
		"messages": [
			{"role":"system","content":"You are helpful."},
			{"role":"user","content":"Weather in Beijing?"},
			{"role":"assistant","content":null,"reasoning_content":"Need the tool.","tool_calls":[
				{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Beijing\"}"}},
				{"id":"call_2","type":"function","function":{"name":"now","arguments":"{}"}}
			]},
			{"role":"tool","tool_call_id":"call_1","content":"Sunny, 25C"},
			{"role":"tool","tool_call_id":"call_2","content":"Error: clock broken"},
			{"role":"user","content":[
				{"type":"text","text":"Also, "},
				{"type":"text","text":"what about this?"},
				{"type":"image_url","image_url":{"url":"https://x/img.png"}},
				{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}
			]},
			{"role":"assistant","content":"Looks like a cat."}
		],
		"tools": [
			{"type":"function","function":{"name":"get_weather","description":"Get weather",
			 "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
			{"type":"function","function":{"name":"now","parameters":{"type":"object","properties":{}}}}
		],
		"tool_choice": "auto",
		"max_completion_tokens": 1024,
		"temperature": 1,
		"top_p": 0.95,
		"stop": ["\n\n"],
		"reasoning_effort": "low",
		"response_format": {"type":"json_object"}
	}`)
}

func TestRequestOptions(t *testing.T) {
	encode := func(c *Client, req *llm.Request, stream bool) map[string]json.RawMessage {
		t.Helper()
		body, err := c.encodeRequest(req, stream)
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]json.RawMessage
		json.Unmarshal(body, &m)
		return m
	}
	req := simpleRequest()
	req.MaxTokens = 10
	req.ToolChoice = &llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "f"}
	m := encode(New(WithMaxTokensField("max_tokens")), req, true)
	if string(m["max_tokens"]) != "10" || m["max_completion_tokens"] != nil {
		t.Errorf("max tokens field: %v", m)
	}
	if string(m["tool_choice"]) != `{"function":{"name":"f"},"type":"function"}` {
		t.Errorf("tool_choice = %s", m["tool_choice"])
	}
	if string(m["stream"]) != "true" || string(m["stream_options"]) != `{"include_usage":true}` {
		t.Errorf("stream fields: %s %s", m["stream"], m["stream_options"])
	}
	if m := encode(New(WithStreamUsage(false)), simpleRequest(), true); m["stream_options"] != nil {
		t.Errorf("stream_options sent when disabled")
	}

	withReasoning := &llm.Request{Model: "m", Messages: []llm.Message{
		llm.UserText("q"),
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{reasoningBlock("hmm"), llm.TextBlock("a")}},
	}}
	if body, _ := New(WithReasoningReplay(false)).encodeRequest(withReasoning, false); strings.Contains(string(body), "reasoning_content") {
		t.Errorf("reasoning replayed when disabled: %s", body)
	}

	for _, bad := range []*llm.Request{
		{Model: "m"},
		{Model: "m", Messages: []llm.Message{{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.ImageURLBlock("u")}}}},
		{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.ToolCallBlock(llm.ToolCall{ID: "x"})}}}},
	} {
		if _, err := New().encodeRequest(bad, false); err == nil {
			t.Errorf("expected error for %+v", bad)
		}
	}
}

func TestGenerateToolCalls(t *testing.T) {
	srv, _ := testutil.Server(t, 200, "application/json", fixture(t, "tool_calls.json"))
	resp, err := newClient(srv).Generate(context.Background(), simpleRequest())
	if err != nil {
		t.Fatal(err)
	}
	want := &llm.Response{
		ID: "chatcmpl-0001", Model: "vendor/deepseek-flash",
		Content: []llm.ContentBlock{
			reasoningBlock("The user wants weather for Beijing and Shanghai. I'll call the tool for both, in parallel."),
			llm.TextBlock("I'll look up the weather for both cities."),
			toolCallBlock("call_aaa", "get_weather", `{"city": "Beijing"}`),
			toolCallBlock("call_bbb", "get_weather", `{"city": "Shanghai"}`),
		},
		StopReason: llm.StopToolUse, ProviderStopReason: "tool_calls",
		Usage: llm.Usage{InputTokens: 302, OutputTokens: 98, ReasoningTokens: 20},
	}
	if !reflect.DeepEqual(resp, want) {
		t.Errorf("got  %s\nwant %s", testutil.Pretty(resp), testutil.Pretty(want))
	}
}

func TestGenerateReasoningOnlyLength(t *testing.T) {
	srv, _ := testutil.Server(t, 200, "application/json", fixture(t, "length_reasoning_only.json"))
	resp, err := newClient(srv).Generate(context.Background(), simpleRequest())
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != llm.StopMaxTokens || resp.Text() != "" || len(resp.Content) != 1 ||
		resp.Content[0].Type != llm.ContentReasoning || resp.Usage.CacheReadTokens != 16 || resp.Usage.ReasoningTokens != 300 {
		t.Errorf("resp = %s", testutil.Pretty(resp))
	}
}

func TestResponseVariants(t *testing.T) {
	tests := []struct {
		name, body string
		check      func(*llm.Response) bool
	}{
		{"content parts", `{"choices":[{"finish_reason":"stop","message":{"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}}]}`,
			func(r *llm.Response) bool { return r.Text() == "ab" && r.StopReason == llm.StopEndTurn }},
		{"refusal", `{"choices":[{"finish_reason":"content_filter","message":{"content":null,"refusal":"no"}}]}`,
			func(r *llm.Response) bool { return r.Text() == "no" && r.StopReason == llm.StopContentFilter }},
		{"empty arguments", `{"choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[{"id":"c","function":{"name":"f","arguments":""}}]}}]}`,
			func(r *llm.Response) bool { return string(r.ToolCalls()[0].Arguments) == "{}" }},
		{"unknown finish", `{"choices":[{"finish_reason":"insufficient_system_resource","message":{"content":"x"}}]}`,
			func(r *llm.Response) bool { return r.StopReason == llm.StopOther }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var r apiResponse
			if err := json.Unmarshal([]byte(tt.body), &r); err != nil {
				t.Fatal(err)
			}
			resp, err := convertResponse(&r)
			if err != nil || !tt.check(resp) {
				t.Errorf("resp = %s, err = %v", testutil.Pretty(resp), err)
			}
		})
	}
}

func TestGenerateErrors(t *testing.T) {
	t.Run("gateway auth error", func(t *testing.T) {
		srv, _ := testutil.Server(t, 401, "application/json",
			`{"error":{"message":"invalid api_key","type":"UserAuthError","code":"2003"},"trace_id":"t","request_id":"t"}`)
		_, err := newClient(srv).Generate(context.Background(), simpleRequest())
		var apiErr *llm.APIError
		if !errors.As(err, &apiErr) || apiErr.StatusCode != 401 || apiErr.Type != "UserAuthError" || apiErr.Code != "2003" || apiErr.Retryable() {
			t.Errorf("err = %#v", err)
		}
	})
	t.Run("rate limit", func(t *testing.T) {
		srv, _ := testutil.Server(t, 429, "application/json", `{"error":{"message":"slow","type":"rate_limit_error"}}`, "Retry-After", "1")
		_, err := newClient(srv).Generate(context.Background(), simpleRequest())
		var apiErr *llm.APIError
		if !errors.As(err, &apiErr) || apiErr.RetryAfter != time.Second || !apiErr.Retryable() {
			t.Errorf("err = %v", err)
		}
	})
	t.Run("error in 200 body", func(t *testing.T) {
		srv, _ := testutil.Server(t, 200, "application/json", `{"error":{"message":"upstream failed","type":"server_error","code":500}}`)
		_, err := newClient(srv).Generate(context.Background(), simpleRequest())
		var apiErr *llm.APIError
		if !errors.As(err, &apiErr) || apiErr.Code != "500" || !apiErr.Retryable() {
			t.Errorf("err = %v", err)
		}
	})
	t.Run("no choices", func(t *testing.T) {
		srv, _ := testutil.Server(t, 200, "application/json", `{"id":"x","choices":[]}`)
		if _, err := newClient(srv).Generate(context.Background(), simpleRequest()); err == nil || !strings.Contains(err.Error(), "no choices") {
			t.Errorf("err = %v", err)
		}
	})
}

func TestStreamToolCalls(t *testing.T) {
	srv, got := testutil.Server(t, 200, "text/event-stream", fixture(t, "stream_tool_calls.sse"))
	s, err := newClient(srv).Stream(context.Background(), simpleRequest())
	if err != nil {
		t.Fatal(err)
	}
	events := testutil.Drain(t, s)
	if got.Header.Get("Accept") != "text/event-stream" {
		t.Errorf("Accept = %q", got.Header.Get("Accept"))
	}

	text := llm.TextBlock("I'll check the weather for both cities.")
	call0 := toolCallBlock("call_aaa", "get_weather", `{"city": "Beijing"}`)
	call1 := toolCallBlock("call_bbb", "get_weather", `{"city": "Shanghai"}`)
	emptyText := llm.TextBlock("")
	want := []llm.StreamEvent{
		{Type: llm.EventStart, Response: &llm.Response{ID: "chatcmpl-0003", Model: "vendor/deepseek-flash"}},
		{Type: llm.EventContentStart, Index: 0, Block: &emptyText},
		{Type: llm.EventTextDelta, Index: 0, Delta: "I"},
		{Type: llm.EventTextDelta, Index: 0, Delta: "'ll check the weather for both"},
		{Type: llm.EventTextDelta, Index: 0, Delta: " cities."},
		{Type: llm.EventContentStart, Index: 1, Block: &llm.ContentBlock{Type: llm.ContentToolCall, ToolCall: &llm.ToolCall{ID: "call_aaa", Name: "get_weather"}}},
		{Type: llm.EventToolCallDelta, Index: 1, Delta: "{"},
		{Type: llm.EventToolCallDelta, Index: 1, Delta: `"city": "Beijing"}`},
		{Type: llm.EventContentStart, Index: 2, Block: &llm.ContentBlock{Type: llm.ContentToolCall, ToolCall: &llm.ToolCall{ID: "call_bbb", Name: "get_weather"}}},
		{Type: llm.EventToolCallDelta, Index: 2, Delta: "{"},
		{Type: llm.EventToolCallDelta, Index: 2, Delta: `"city": "Shanghai"}`},
		{Type: llm.EventContentStop, Index: 0, Block: &text},
		{Type: llm.EventContentStop, Index: 1, Block: &call0},
		{Type: llm.EventContentStop, Index: 2, Block: &call1},
		{Type: llm.EventDone, Response: &llm.Response{
			ID: "chatcmpl-0003", Model: "vendor/deepseek-flash",
			Content:    []llm.ContentBlock{text, call0, call1},
			StopReason: llm.StopToolUse, ProviderStopReason: "tool_calls",
			Usage: llm.Usage{InputTokens: 302, OutputTokens: 77},
		}},
	}
	testutil.AssertEvents(t, events, want)
}

func TestStreamReasoning(t *testing.T) {
	srv, _ := testutil.Server(t, 200, "text/event-stream", fixture(t, "stream_reasoning.sse"))
	s, err := newClient(srv).Stream(context.Background(), simpleRequest())
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.Collect(s)
	if err != nil {
		t.Fatal(err)
	}
	want := []llm.ContentBlock{reasoningBlock("We add 2 and 2."), llm.TextBlock("4")}
	if !reflect.DeepEqual(resp.Content, want) || resp.StopReason != llm.StopEndTurn || resp.Usage.ReasoningTokens != 25 {
		t.Errorf("resp = %s", testutil.Pretty(resp))
	}
}

func TestStreamVariants(t *testing.T) {
	chunk := func(delta string, extra string) string {
		return `data: {"id":"c","model":"m","choices":[{"index":0,"delta":` + delta + extra + `}]}` + "\n\n"
	}
	t.Run("no DONE after finish_reason", func(t *testing.T) {
		srv, _ := testutil.Server(t, 200, "text/event-stream", chunk(`{"content":"hi"}`, "")+chunk(`{}`, `,"finish_reason":"stop"`))
		s, _ := newClient(srv).Stream(context.Background(), simpleRequest())
		resp, err := llm.Collect(s)
		if err != nil || resp.Text() != "hi" {
			t.Errorf("resp=%v err=%v", resp, err)
		}
	})
	t.Run("tool calls without index", func(t *testing.T) {
		body := chunk(`{"tool_calls":[{"id":"a","function":{"name":"f","arguments":"{\"x\":"}}]}`, "") +
			chunk(`{"tool_calls":[{"function":{"arguments":"1}"}}]}`, "") +
			chunk(`{"tool_calls":[{"id":"b","function":{"name":"g","arguments":"{}"}}]}`, `,"finish_reason":"tool_calls"`) +
			"data: [DONE]\n\n"
		srv, _ := testutil.Server(t, 200, "text/event-stream", body)
		s, _ := newClient(srv).Stream(context.Background(), simpleRequest())
		resp, err := llm.Collect(s)
		if err != nil {
			t.Fatal(err)
		}
		calls := resp.ToolCalls()
		if len(calls) != 2 || string(calls[0].Arguments) != `{"x":1}` || calls[1].Name != "g" {
			t.Errorf("calls = %+v", calls)
		}
	})
	t.Run("usage chunk with empty choices", func(t *testing.T) {
		body := chunk(`{"content":"a"}`, `,"finish_reason":"stop"`) +
			`data: {"id":"c","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}` + "\n\ndata: [DONE]\n\n"
		srv, _ := testutil.Server(t, 200, "text/event-stream", body)
		s, _ := newClient(srv).Stream(context.Background(), simpleRequest())
		resp, err := llm.Collect(s)
		if err != nil || resp.Usage.InputTokens != 3 || resp.Usage.OutputTokens != 1 {
			t.Errorf("resp=%+v err=%v", resp, err)
		}
	})
}

func TestStreamErrors(t *testing.T) {
	t.Run("http error", func(t *testing.T) {
		srv, _ := testutil.Server(t, 401, "application/json", `{"error":{"message":"model not found","type":"UserModelNotFound","code":"2004"}}`)
		_, err := newClient(srv).Stream(context.Background(), simpleRequest())
		var apiErr *llm.APIError
		if !errors.As(err, &apiErr) || apiErr.Type != "UserModelNotFound" {
			t.Errorf("err = %v", err)
		}
	})
	cases := []struct {
		name  string
		body  string
		check func(error) bool
	}{
		{"error chunk", `data: {"id":"c","choices":[{"index":0,"delta":{"content":"a"}}]}` + "\n\n" +
			`data: {"error":{"message":"overloaded","type":"server_error"}}` + "\n\n",
			func(err error) bool {
				var e *llm.APIError
				return errors.As(err, &e) && e.Message == "overloaded" && e.Retryable()
			}},
		{"truncated", `data: {"id":"c","choices":[{"index":0,"delta":{"content":"a"}}]}` + "\n\n",
			func(err error) bool { return errors.Is(err, llm.ErrIncompleteStream) }},
		{"malformed", "data: {nope\n\n", func(err error) bool { return err != nil && strings.Contains(err.Error(), "decode") }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := testutil.Server(t, 200, "text/event-stream", tc.body)
			s, err := newClient(srv).Stream(context.Background(), simpleRequest())
			if err != nil {
				t.Fatal(err)
			}
			if _, err := llm.Collect(s); !tc.check(err) {
				t.Errorf("err = %v", err)
			}
		})
	}
}

func TestLegacyReasoningProviderIsReplayed(t *testing.T) {
	if ProviderName != "openai/chat-completions" {
		t.Fatalf("ProviderName = %q", ProviderName)
	}
	body, err := New().encodeRequest(&llm.Request{Model: "m", Messages: []llm.Message{
		llm.UserText("q"),
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: "chatcompletions", Text: "legacy"}},
			{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: "openai", Text: "foreign"}},
			llm.TextBlock("a"),
		}},
	}}, false)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), `"reasoning_content":"legacy"`) || strings.Contains(string(body), "foreign") {
		t.Errorf("body = %s", body)
	}
}

func TestErrorProviderField(t *testing.T) {
	srv, _ := testutil.Server(t, 500, "application/json", `{"error":{"message":"x"}}`)
	_, err := newClient(srv).Generate(context.Background(), simpleRequest())
	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) || apiErr.Provider != "openai/chat-completions" || !strings.HasPrefix(err.Error(), "openai/chat-completions: HTTP 500") {
		t.Errorf("err = %v", err)
	}
}

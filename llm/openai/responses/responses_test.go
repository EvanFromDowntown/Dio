package responses

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/EvanFromDowntown/Dio/internal/testutil"
	"github.com/EvanFromDowntown/Dio/llm"
)

var (
	newServer     = testutil.Server
	assertJSON    = testutil.AssertJSON
	sseBody       = testutil.SSE
	collectEvents = testutil.Drain
)

func newClient(srv *httptest.Server, opts ...Option) *Client {
	return New(append([]Option{WithBaseURL(srv.URL + "/v1/"), WithAPIKey("sk-test"), WithHTTPClient(srv.Client())}, opts...)...)
}

func fullRequest() *llm.Request {
	return &llm.Request{
		Model:  "gpt-test",
		System: "be brief",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{
				llm.TextBlock("weather?"),
				llm.ImageURLBlock("https://x/img.png"),
				llm.ImageDataBlock("image/png", "AAAA"),
			}},
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: ProviderName, ID: "rs_1", Text: "think", Encrypted: "enc"}},
				{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: "anthropic", Text: "foreign", Signature: "sig"}},
				llm.TextBlock("checking"),
				llm.ToolCallBlock(llm.ToolCall{ID: "call_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"SF"}`)}),
				llm.ToolCallBlock(llm.ToolCall{ID: "call_2", Name: "now"}),
			}},
			llm.ToolResults(
				llm.ToolResult{ToolCallID: "call_1", Content: "sunny"},
				llm.ToolResult{ToolCallID: "call_2", Content: "", IsError: true},
			),
		},
		Tools: []llm.Tool{
			{Name: "get_weather", Description: "Get weather", InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)},
			{Name: "now"},
		},
		ToolChoice:  &llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "get_weather"},
		MaxTokens:   256,
		Temperature: llm.Ptr(0.2),
		TopP:        llm.Ptr(0.9),
		Reasoning:   &llm.ReasoningConfig{Effort: "low", Summary: "auto"},
		Extra:       map[string]any{"store": false, "include": []string{"reasoning.encrypted_content"}},
	}
}

func TestGenerateRequestSerialization(t *testing.T) {
	srv, got := newServer(t, 200, "application/json", `{"id":"resp_1","status":"completed","output":[]}`)
	c := newClient(srv, WithHeader("X-Gateway", "dio"))
	if _, err := c.Generate(context.Background(), fullRequest()); err != nil {
		t.Fatal(err)
	}
	if got.Method != http.MethodPost || got.Path != "/v1/responses" {
		t.Errorf("request line: %s %s", got.Method, got.Path)
	}
	if h := got.Header.Get("Authorization"); h != "Bearer sk-test" {
		t.Errorf("Authorization = %q", h)
	}
	if h := got.Header.Get("X-Gateway"); h != "dio" {
		t.Errorf("X-Gateway = %q", h)
	}
	if h := got.Header.Get("Content-Type"); h != "application/json" {
		t.Errorf("Content-Type = %q", h)
	}
	assertJSON(t, got.Body, `{
		"model": "gpt-test",
		"instructions": "be brief",
		"input": [
			{"type":"message","role":"user","content":[
				{"type":"input_text","text":"weather?"},
				{"type":"input_image","image_url":"https://x/img.png"},
				{"type":"input_image","image_url":"data:image/png;base64,AAAA"}
			]},
			{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"think"}],"encrypted_content":"enc"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"checking"}]},
			{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"SF\"}"},
			{"type":"function_call","call_id":"call_2","name":"now","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_1","output":"sunny"},
			{"type":"function_call_output","call_id":"call_2","output":""}
		],
		"tools": [
			{"type":"function","name":"get_weather","description":"Get weather","strict":false,
			 "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},
			{"type":"function","name":"now","strict":false,"parameters":{"type":"object","properties":{}}}
		],
		"tool_choice": {"type":"function","name":"get_weather"},
		"max_output_tokens": 256,
		"temperature": 0.2,
		"top_p": 0.9,
		"reasoning": {"effort":"low","summary":"auto"},
		"store": false,
		"include": ["reasoning.encrypted_content"]
	}`)
}

func TestToolChoiceModes(t *testing.T) {
	for mode, want := range map[llm.ToolChoiceMode]string{
		llm.ToolChoiceAuto: `"auto"`, llm.ToolChoiceNone: `"none"`, llm.ToolChoiceRequired: `"required"`,
	} {
		body, err := encodeRequest(&llm.Request{
			Model: "m", Messages: []llm.Message{llm.UserText("x")}, ToolChoice: &llm.ToolChoice{Mode: mode},
		}, false)
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]json.RawMessage
		json.Unmarshal(body, &m)
		if string(m["tool_choice"]) != want {
			t.Errorf("%s: tool_choice = %s", mode, m["tool_choice"])
		}
	}
}

func TestRequestValidationErrors(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { calls.Add(1) }))
	defer srv.Close()
	c := newClient(srv)
	bad := []*llm.Request{
		{Messages: []llm.Message{llm.UserText("x")}},
		{Model: "m", Messages: []llm.Message{llm.UserText("x")}, StopSequences: []string{"END"}},
		{Model: "m", Messages: []llm.Message{{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.ImageURLBlock("u")}}}},
		{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.ContentBlock{{Type: llm.ContentImage, Image: &llm.Image{Data: "x"}}}}}},
	}
	for i, r := range bad {
		if _, err := c.Generate(context.Background(), r); err == nil {
			t.Errorf("case %d: expected error", i)
		}
	}
	if calls.Load() != 0 {
		t.Errorf("invalid requests reached the server %d times", calls.Load())
	}
}

func TestGenerateResponseParsing(t *testing.T) {
	srv, _ := newServer(t, 200, "application/json", `{
		"id": "resp_1", "object": "response", "status": "completed", "model": "gpt-test-2026",
		"output": [
			{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"a"},{"type":"summary_text","text":"b"}],"encrypted_content":"enc"},
			{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[
				{"type":"output_text","text":"Let me check.","annotations":[]}
			]},
			{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"SF\"}","status":"completed"},
			{"type":"web_search_call","id":"ws_1","status":"completed"}
		],
		"usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 40},
		          "output_tokens": 30, "output_tokens_details": {"reasoning_tokens": 12}, "total_tokens": 130}
	}`)
	resp, err := newClient(srv).Generate(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("x")}})
	if err != nil {
		t.Fatal(err)
	}
	want := &llm.Response{
		ID: "resp_1", Model: "gpt-test-2026",
		Content: []llm.ContentBlock{
			{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: ProviderName, ID: "rs_1", Text: "a\n\nb", Encrypted: "enc"}},
			llm.TextBlock("Let me check."),
			llm.ToolCallBlock(llm.ToolCall{ID: "call_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"SF"}`)}),
		},
		StopReason: llm.StopToolUse, ProviderStopReason: "completed",
		Usage: llm.Usage{InputTokens: 100, OutputTokens: 30, CacheReadTokens: 40, ReasoningTokens: 12},
	}
	if !reflect.DeepEqual(resp, want) {
		t.Errorf("got  %+v\nwant %+v", resp, want)
	}
}

// Gateways serving open-weight models return raw reasoning as content parts
// (output_text when non-streaming, reasoning_text when streaming) with an
// empty summary.
func TestReasoningContentParts(t *testing.T) {
	var r apiResponse
	json.Unmarshal([]byte(`{"status":"completed","output":[
		{"type":"reasoning","id":"rs_-1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"think","annotations":[]}]},
		{"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}`), &r)
	resp, err := convertResponse(&r)
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Content[0].Reasoning; got == nil || got.Text != "think" || got.ID != "rs_-1" {
		t.Errorf("reasoning = %+v", got)
	}

	srv, _ := newServer(t, 200, "text/event-stream", "data: "+strings.Join([]string{
		`{"type":"response.output_item.added","output_index":0,"item":{"summary":[],"id":"msg_r","type":"reasoning","content":[]}}`,
		`{"type":"response.reasoning_text.delta","delta":"User","output_index":0,"content_index":0,"item_id":"msg_r"}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"summary":[],"id":"msg_r","type":"reasoning","content":[{"type":"reasoning_text","text":"User asks"}]}}`,
		`{"type":"response.completed","response":{"id":"r","status":"completed","output":[{"id":"msg_r","summary":[],"type":"reasoning","content":[{"text":"User asks","type":"reasoning_text"}]}]}}`,
	}, "\n\ndata: ")+"\n\n")
	s, err := newClient(srv).Stream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("x")}})
	if err != nil {
		t.Fatal(err)
	}
	events := collectEvents(t, s)
	if events[1].Type != llm.EventReasoningDelta || events[1].Delta != "User" ||
		events[2].Block.Reasoning.Text != "User asks" || events[3].Response.Content[0].Reasoning.Text != "User asks" {
		t.Errorf("events = %s", testutil.Pretty(events))
	}
}

func TestStopReasons(t *testing.T) {
	tests := []struct {
		body string
		want llm.StopReason
		raw  string
	}{
		{`{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}]}`, llm.StopEndTurn, "completed"},
		{`{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[]}`, llm.StopMaxTokens, "max_output_tokens"},
		{`{"status":"incomplete","incomplete_details":{"reason":"content_filter"},"output":[]}`, llm.StopContentFilter, "content_filter"},
		{`{"status":"cancelled","output":[]}`, llm.StopOther, "cancelled"},
	}
	for _, tt := range tests {
		var r apiResponse
		if err := json.Unmarshal([]byte(tt.body), &r); err != nil {
			t.Fatal(err)
		}
		resp, err := convertResponse(&r)
		if err != nil {
			t.Fatal(err)
		}
		if resp.StopReason != tt.want || resp.ProviderStopReason != tt.raw {
			t.Errorf("%s: got %q/%q", tt.body, resp.StopReason, resp.ProviderStopReason)
		}
	}
}

func TestGenerateErrors(t *testing.T) {
	req := &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("x")}}

	t.Run("rate limit", func(t *testing.T) {
		srv, _ := newServer(t, 429, "application/json",
			`{"error":{"message":"Rate limit reached","type":"requests","code":"rate_limit_exceeded","param":null}}`,
			"Retry-After", "2")
		_, err := newClient(srv).Generate(context.Background(), req)
		var apiErr *llm.APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("err = %v", err)
		}
		if apiErr.StatusCode != 429 || apiErr.Code != "rate_limit_exceeded" || apiErr.Message != "Rate limit reached" ||
			apiErr.RetryAfter != 2*time.Second || !apiErr.Retryable() || apiErr.Provider != ProviderName {
			t.Errorf("unexpected error: %+v", apiErr)
		}
	})

	t.Run("auth with non-JSON body", func(t *testing.T) {
		srv, _ := newServer(t, 401, "text/plain", "unauthorized")
		_, err := newClient(srv).Generate(context.Background(), req)
		var apiErr *llm.APIError
		if !errors.As(err, &apiErr) || apiErr.StatusCode != 401 || apiErr.Message != "unauthorized" || apiErr.Retryable() {
			t.Errorf("err = %#v", err)
		}
	})

	t.Run("failed response status", func(t *testing.T) {
		srv, _ := newServer(t, 200, "application/json",
			`{"id":"r","status":"failed","error":{"code":"server_error","message":"oops"},"output":[]}`)
		_, err := newClient(srv).Generate(context.Background(), req)
		var apiErr *llm.APIError
		if !errors.As(err, &apiErr) || apiErr.Code != "server_error" || !apiErr.Retryable() {
			t.Errorf("err = %v", err)
		}
	})

	t.Run("malformed body", func(t *testing.T) {
		srv, _ := newServer(t, 200, "application/json", `{"id":`)
		if _, err := newClient(srv).Generate(context.Background(), req); err == nil || !strings.Contains(err.Error(), "decode") {
			t.Errorf("err = %v", err)
		}
	})

	t.Run("context canceled", func(t *testing.T) {
		srv, _ := newServer(t, 200, "application/json", `{}`)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := newClient(srv).Generate(ctx, req); !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v", err)
		}
	})
}

type countingTransport struct {
	n    atomic.Int32
	base http.RoundTripper
}

func (c *countingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	c.n.Add(1)
	return c.base.RoundTrip(r)
}

func TestHTTPClientInjection(t *testing.T) {
	srv, _ := newServer(t, 200, "application/json", `{"status":"completed","output":[]}`)
	rt := &countingTransport{base: srv.Client().Transport}
	c := New(WithBaseURL(srv.URL), WithHTTPClient(&http.Client{Transport: rt}))
	if _, err := c.Generate(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("x")}}); err != nil {
		t.Fatal(err)
	}
	if rt.n.Load() != 1 {
		t.Errorf("custom transport used %d times", rt.n.Load())
	}
}

func TestStream(t *testing.T) {
	completed := `{"id":"resp_1","status":"completed","model":"gpt-test","output":[
		{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Plan"},{"type":"summary_text","text":"More"}]},
		{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"Hi there"}]},
		{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"SF\"}"}
	],"usage":{"input_tokens":10,"output_tokens":5}}`
	body := sseBody(
		`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","status":"in_progress","model":"gpt-test","output":[]}}`,
		`{"type":"response.in_progress","response":{"id":"resp_1"}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}`,
		`{"type":"response.reasoning_summary_part.added","item_id":"rs_1","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}`,
		`{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"Plan"}`,
		`{"type":"response.reasoning_summary_part.added","item_id":"rs_1","output_index":0,"summary_index":1,"part":{"type":"summary_text","text":""}}`,
		`{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":1,"delta":"More"}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Plan"},{"type":"summary_text","text":"More"}]}}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}`,
		`{"type":"response.content_part.added","item_id":"msg_1","output_index":1,"content_index":0,"part":{"type":"output_text","text":""}}`,
		`{"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":0,"delta":"Hi"}`,
		`{"type":"response.output_text.delta","item_id":"msg_1","output_index":1,"content_index":0,"delta":" there"}`,
		`{"type":"response.output_text.done","item_id":"msg_1","output_index":1,"content_index":0,"text":"Hi there"}`,
		`{"type":"response.content_part.done","item_id":"msg_1","output_index":1,"content_index":0,"part":{"type":"output_text","text":"Hi there"}}`,
		`{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"Hi there"}]}}`,
		`{"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":2,"delta":"{\"city\":"}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":2,"delta":"\"SF\"}"}`,
		`{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":2,"arguments":"{\"city\":\"SF\"}"}`,
		`{"type":"response.output_item.done","output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"SF\"}"}}`,
		`{"type":"response.completed","response":`+completed+`}`,
	)
	srv, got := newServer(t, 200, "text/event-stream", body)
	s, err := newClient(srv).Stream(context.Background(), &llm.Request{Model: "gpt-test", Messages: []llm.Message{llm.UserText("x")}})
	if err != nil {
		t.Fatal(err)
	}
	events := collectEvents(t, s)

	if got.Header.Get("Accept") != "text/event-stream" {
		t.Errorf("Accept = %q", got.Header.Get("Accept"))
	}
	var req map[string]any
	json.Unmarshal(got.Body, &req)
	if req["stream"] != true {
		t.Errorf("stream flag not set: %s", got.Body)
	}

	reasoning := llm.ContentBlock{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: ProviderName, ID: "rs_1", Text: "Plan\n\nMore"}}
	toolCall := llm.ToolCallBlock(llm.ToolCall{ID: "call_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"SF"}`)})
	wantResp := &llm.Response{
		ID: "resp_1", Model: "gpt-test",
		Content:    []llm.ContentBlock{reasoning, llm.TextBlock("Hi there"), toolCall},
		StopReason: llm.StopToolUse, ProviderStopReason: "completed",
		Usage: llm.Usage{InputTokens: 10, OutputTokens: 5},
	}
	empty := llm.TextBlock("")
	text := llm.TextBlock("Hi there")
	want := []llm.StreamEvent{
		{Type: llm.EventStart, Response: &llm.Response{ID: "resp_1", Model: "gpt-test"}},
		{Type: llm.EventContentStart, Index: 0, Block: &llm.ContentBlock{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: ProviderName, ID: "rs_1"}}},
		{Type: llm.EventReasoningDelta, Index: 0, Delta: "Plan"},
		{Type: llm.EventReasoningDelta, Index: 0, Delta: "\n\n"},
		{Type: llm.EventReasoningDelta, Index: 0, Delta: "More"},
		{Type: llm.EventContentStop, Index: 0, Block: &reasoning},
		{Type: llm.EventContentStart, Index: 1, Block: &empty},
		{Type: llm.EventTextDelta, Index: 1, Delta: "Hi"},
		{Type: llm.EventTextDelta, Index: 1, Delta: " there"},
		{Type: llm.EventContentStop, Index: 1, Block: &text},
		{Type: llm.EventContentStart, Index: 2, Block: &llm.ContentBlock{Type: llm.ContentToolCall, ToolCall: &llm.ToolCall{ID: "call_1", Name: "get_weather"}}},
		{Type: llm.EventToolCallDelta, Index: 2, Delta: `{"city":`},
		{Type: llm.EventToolCallDelta, Index: 2, Delta: `"SF"}`},
		{Type: llm.EventContentStop, Index: 2, Block: &toolCall},
		{Type: llm.EventDone, Response: wantResp},
	}
	testutil.AssertEvents(t, events, want)
}

func TestStreamCollectIncomplete(t *testing.T) {
	srv, _ := newServer(t, 200, "text/event-stream", sseBody(
		`{"type":"response.incomplete","response":{"id":"r","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[
			{"type":"message","content":[{"type":"output_text","text":"cut"}]}]}}`,
	))
	s, err := newClient(srv).Stream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("x")}})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := llm.Collect(s)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != llm.StopMaxTokens || resp.Text() != "cut" {
		t.Errorf("resp = %+v", resp)
	}
}

func TestStreamErrors(t *testing.T) {
	req := &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("x")}}
	start := `{"type":"response.created","response":{"id":"r"}}`

	t.Run("http error before stream", func(t *testing.T) {
		srv, _ := newServer(t, 500, "application/json", `{"error":{"message":"internal","type":"server_error"}}`)
		_, err := newClient(srv).Stream(context.Background(), req)
		var apiErr *llm.APIError
		if !errors.As(err, &apiErr) || apiErr.StatusCode != 500 || !apiErr.Retryable() {
			t.Errorf("err = %v", err)
		}
	})

	cases := []struct {
		name  string
		body  string
		check func(error) bool
	}{
		{"error event", sseBody(start, `{"type":"error","code":"rate_limit_exceeded","message":"slow down","param":null}`),
			func(err error) bool {
				var e *llm.APIError
				return errors.As(err, &e) && e.Code == "rate_limit_exceeded" && e.Message == "slow down" && e.Retryable()
			}},
		{"response.failed", sseBody(start, `{"type":"response.failed","response":{"id":"r","status":"failed","error":{"code":"server_error","message":"x"}}}`),
			func(err error) bool { var e *llm.APIError; return errors.As(err, &e) && e.Code == "server_error" }},
		{"truncated", sseBody(start), func(err error) bool { return errors.Is(err, llm.ErrIncompleteStream) }},
		{"malformed event", "data: {not json\n\n", func(err error) bool { return err != nil && strings.Contains(err.Error(), "decode") }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := newServer(t, 200, "text/event-stream", tc.body)
			s, err := newClient(srv).Stream(context.Background(), req)
			if err != nil {
				t.Fatal(err)
			}
			_, err = llm.Collect(s)
			if !tc.check(err) {
				t.Errorf("err = %v", err)
			}
		})
	}
}

func TestLegacyReasoningProviderIsReplayed(t *testing.T) {
	if ProviderName != "openai/responses" {
		t.Fatalf("ProviderName = %q", ProviderName)
	}
	body, err := encodeRequest(&llm.Request{Model: "m", Messages: []llm.Message{
		llm.UserText("q"),
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: "openai", ID: "rs_old", Text: "legacy"}},
			{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: "chatcompletions", ID: "rs_x", Text: "foreign"}},
		}},
	}}, false)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), `"id":"rs_old"`) || strings.Contains(string(body), "rs_x") {
		t.Errorf("body = %s", body)
	}
}

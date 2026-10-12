package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/EvanFromDowntown/Dio/internal/testutil"
	"github.com/EvanFromDowntown/Dio/llm"
)

func newClient(srv *httptest.Server, opts ...Option) *Client {
	return New(append([]Option{WithBaseURL(srv.URL + "/"), WithAPIKey("sk-ant-test"), WithHTTPClient(srv.Client())}, opts...)...)
}

func simpleRequest() *llm.Request {
	return &llm.Request{Model: "claude-test", Messages: []llm.Message{llm.UserText("x")}}
}

const okResponse = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`

func TestGenerateRequestSerialization(t *testing.T) {
	srv, got := testutil.Server(t, 200, "application/json", okResponse)
	c := newClient(srv, WithHeader("anthropic-beta", "test-beta"))
	req := &llm.Request{
		Model:  "claude-test",
		System: "be brief",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.ContentBlock{
				llm.TextBlock("weather?"),
				llm.ImageDataBlock("image/png", "AAAA"),
				llm.ImageURLBlock("https://x/img.png"),
			}},
			{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
				{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: ProviderName, Text: "think", Signature: "sig"}},
				{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: ProviderName, Encrypted: "redacted"}},
				{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: "openai", ID: "rs_1", Text: "foreign"}},
				llm.TextBlock("checking"),
				llm.ToolCallBlock(llm.ToolCall{ID: "toolu_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"SF"}`)}),
				llm.ToolCallBlock(llm.ToolCall{ID: "toolu_2", Name: "now"}),
			}},
			llm.ToolResults(
				llm.ToolResult{ToolCallID: "toolu_1", Content: "sunny"},
				llm.ToolResult{ToolCallID: "toolu_2", Content: "clock broken", IsError: true},
			),
		},
		Tools: []llm.Tool{
			{Name: "get_weather", Description: "Get weather", InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)},
			{Name: "now"},
		},
		ToolChoice:    &llm.ToolChoice{Mode: llm.ToolChoiceRequired},
		Temperature:   llm.Ptr(1.0),
		TopP:          llm.Ptr(0.95),
		StopSequences: []string{"END"},
		Reasoning:     &llm.ReasoningConfig{Effort: "medium"},
		Extra:         map[string]any{"metadata": map[string]string{"user_id": "u1"}},
	}
	if _, err := c.Generate(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if got.Method != http.MethodPost || got.Path != "/v1/messages" {
		t.Errorf("request line: %s %s", got.Method, got.Path)
	}
	for k, want := range map[string]string{
		"x-api-key":         "sk-ant-test",
		"anthropic-version": APIVersion,
		"anthropic-beta":    "test-beta",
		"Content-Type":      "application/json",
		"Authorization":     "",
	} {
		if h := got.Header.Get(k); h != want {
			t.Errorf("%s = %q, want %q", k, h, want)
		}
	}
	testutil.AssertJSON(t, got.Body, `{
		"model": "claude-test",
		"max_tokens": 4096,
		"system": "be brief",
		"messages": [
			{"role":"user","content":[
				{"type":"text","text":"weather?"},
				{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}},
				{"type":"image","source":{"type":"url","url":"https://x/img.png"}}
			]},
			{"role":"assistant","content":[
				{"type":"thinking","thinking":"think","signature":"sig"},
				{"type":"redacted_thinking","data":"redacted"},
				{"type":"text","text":"checking"},
				{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"SF"}},
				{"type":"tool_use","id":"toolu_2","name":"now","input":{}}
			]},
			{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"},
				{"type":"tool_result","tool_use_id":"toolu_2","content":"clock broken","is_error":true}
			]}
		],
		"tools": [
			{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}},
			{"name":"now","input_schema":{"type":"object","properties":{}}}
		],
		"tool_choice": {"type":"any"},
		"temperature": 1,
		"top_p": 0.95,
		"stop_sequences": ["END"],
		"thinking": {"type":"enabled","budget_tokens":4096},
		"metadata": {"user_id":"u1"}
	}`)
}

func TestRequestOptions(t *testing.T) {
	encode := func(req *llm.Request) map[string]json.RawMessage {
		t.Helper()
		body, err := encodeRequest(req, 1000, false)
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]json.RawMessage
		json.Unmarshal(body, &m)
		return m
	}
	base := simpleRequest()
	if got := string(encode(base)["max_tokens"]); got != "1000" {
		t.Errorf("default max_tokens = %s", got)
	}
	base.MaxTokens = 50
	if got := string(encode(base)["max_tokens"]); got != "50" {
		t.Errorf("explicit max_tokens = %s", got)
	}
	for mode, want := range map[llm.ToolChoiceMode]string{
		llm.ToolChoiceAuto: `{"type":"auto"}`, llm.ToolChoiceNone: `{"type":"none"}`,
	} {
		r := simpleRequest()
		r.ToolChoice = &llm.ToolChoice{Mode: mode}
		if got := string(encode(r)["tool_choice"]); got != want {
			t.Errorf("%s: tool_choice = %s", mode, got)
		}
	}
	r := simpleRequest()
	r.ToolChoice = &llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "f"}
	r.Reasoning = &llm.ReasoningConfig{BudgetTokens: 2048, Effort: "high"}
	m := encode(r)
	if got := string(m["tool_choice"]); got != `{"type":"tool","name":"f"}` {
		t.Errorf("tool_choice = %s", got)
	}
	if got := string(m["thinking"]); got != `{"type":"enabled","budget_tokens":2048}` {
		t.Errorf("thinking = %s", got)
	}

	r = simpleRequest()
	r.Reasoning = &llm.ReasoningConfig{Effort: "extreme"}
	if _, err := encodeRequest(r, 1000, false); err == nil {
		t.Error("unknown effort without budget should fail")
	}
}

func TestAuthToken(t *testing.T) {
	srv, got := testutil.Server(t, 200, "application/json", okResponse)
	c := New(WithBaseURL(srv.URL), WithAuthToken("gw-token"), WithHTTPClient(srv.Client()))
	if _, err := c.Generate(context.Background(), simpleRequest()); err != nil {
		t.Fatal(err)
	}
	if h := got.Header.Get("Authorization"); h != "Bearer gw-token" {
		t.Errorf("Authorization = %q", h)
	}
	if h := got.Header.Get("x-api-key"); h != "" {
		t.Errorf("x-api-key = %q", h)
	}
}

func TestGenerateResponseParsing(t *testing.T) {
	srv, _ := testutil.Server(t, 200, "application/json", `{
		"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test-2026",
		"content": [
			{"type":"thinking","thinking":"Let me think","signature":"sig"},
			{"type":"redacted_thinking","data":"opaque"},
			{"type":"text","text":"Checking weather."},
			{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"SF"}},
			{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}
		],
		"stop_reason": "tool_use", "stop_sequence": null,
		"usage": {"input_tokens": 10, "cache_creation_input_tokens": 20, "cache_read_input_tokens": 30, "output_tokens": 40}
	}`)
	resp, err := newClient(srv).Generate(context.Background(), simpleRequest())
	if err != nil {
		t.Fatal(err)
	}
	want := &llm.Response{
		ID: "msg_1", Model: "claude-test-2026",
		Content: []llm.ContentBlock{
			{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: ProviderName, Text: "Let me think", Signature: "sig"}},
			{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: ProviderName, Encrypted: "opaque"}},
			llm.TextBlock("Checking weather."),
			llm.ToolCallBlock(llm.ToolCall{ID: "toolu_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"SF"}`)}),
		},
		StopReason: llm.StopToolUse, ProviderStopReason: "tool_use",
		Usage: llm.Usage{InputTokens: 60, OutputTokens: 40, CacheReadTokens: 30, CacheWriteTokens: 20},
	}
	if !reflect.DeepEqual(resp, want) {
		t.Errorf("got  %s\nwant %s", testutil.Pretty(resp), testutil.Pretty(want))
	}

	for raw, want := range map[string]llm.StopReason{
		"end_turn": llm.StopEndTurn, "max_tokens": llm.StopMaxTokens, "stop_sequence": llm.StopSequence,
		"tool_use": llm.StopToolUse, "refusal": llm.StopContentFilter, "pause_turn": llm.StopOther,
	} {
		if got := convertStopReason(raw); got != want {
			t.Errorf("stop reason %s = %s", raw, got)
		}
	}
}

func TestGenerateErrors(t *testing.T) {
	t.Run("overloaded", func(t *testing.T) {
		srv, _ := testutil.Server(t, 529, "application/json",
			`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)
		_, err := newClient(srv).Generate(context.Background(), simpleRequest())
		var apiErr *llm.APIError
		if !errors.As(err, &apiErr) || apiErr.StatusCode != 529 || apiErr.Type != "overloaded_error" || !apiErr.Retryable() {
			t.Errorf("err = %v", err)
		}
	})

	t.Run("rate limit with retry-after", func(t *testing.T) {
		srv, _ := testutil.Server(t, 429, "application/json",
			`{"type":"error","error":{"type":"rate_limit_error","message":"slow"}}`, "Retry-After", "7")
		_, err := newClient(srv).Generate(context.Background(), simpleRequest())
		var apiErr *llm.APIError
		if !errors.As(err, &apiErr) || apiErr.RetryAfter != 7*time.Second || !llm.IsRetryable(err) {
			t.Errorf("err = %v", err)
		}
	})

	t.Run("invalid request", func(t *testing.T) {
		srv, _ := testutil.Server(t, 400, "application/json",
			`{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long"}}`)
		_, err := newClient(srv).Generate(context.Background(), simpleRequest())
		if err == nil || llm.IsRetryable(err) || !strings.Contains(err.Error(), "prompt is too long") {
			t.Errorf("err = %v", err)
		}
	})

	t.Run("validation happens before the request", func(t *testing.T) {
		if _, err := New().Generate(context.Background(), &llm.Request{Model: "m"}); err == nil {
			t.Error("expected validation error")
		}
	})
}

func TestStream(t *testing.T) {
	body := testutil.SSE(
		`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,
			"usage":{"input_tokens":10,"cache_read_input_tokens":5,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Hmm"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"ping"}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"server_tool_use","id":"srv_1","name":"web_search","input":{}}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Hi"}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":" there"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`,
		`{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":""}}`,
		`{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`,
		`{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"\"SF\"}"}}`,
		`{"type":"content_block_stop","index":3}`,
		`{"type":"content_block_start","index":4,"content_block":{"type":"tool_use","id":"toolu_2","name":"now","input":{}}}`,
		`{"type":"content_block_stop","index":4}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":42}}`,
		`{"type":"message_stop"}`,
	)
	srv, got := testutil.Server(t, 200, "text/event-stream", body)
	s, err := newClient(srv).Stream(context.Background(), simpleRequest())
	if err != nil {
		t.Fatal(err)
	}
	events := testutil.Drain(t, s)

	if got.Header.Get("Accept") != "text/event-stream" {
		t.Errorf("Accept = %q", got.Header.Get("Accept"))
	}
	var req map[string]any
	json.Unmarshal(got.Body, &req)
	if req["stream"] != true {
		t.Errorf("stream flag not set: %s", got.Body)
	}

	thinking := llm.ContentBlock{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: ProviderName, Text: "Hmm", Signature: "sig"}}
	text := llm.TextBlock("Hi there")
	call1 := llm.ToolCallBlock(llm.ToolCall{ID: "toolu_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"SF"}`)})
	call2 := llm.ToolCallBlock(llm.ToolCall{ID: "toolu_2", Name: "now", Arguments: json.RawMessage(`{}`)})
	emptyText := llm.TextBlock("")
	want := []llm.StreamEvent{
		{Type: llm.EventStart, Response: &llm.Response{ID: "msg_1", Model: "claude-test"}},
		{Type: llm.EventContentStart, Index: 0, Block: &llm.ContentBlock{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: ProviderName}}},
		{Type: llm.EventReasoningDelta, Index: 0, Delta: "Hmm"},
		{Type: llm.EventContentStop, Index: 0, Block: &thinking},
		{Type: llm.EventContentStart, Index: 1, Block: &emptyText},
		{Type: llm.EventTextDelta, Index: 1, Delta: "Hi"},
		{Type: llm.EventTextDelta, Index: 1, Delta: " there"},
		{Type: llm.EventContentStop, Index: 1, Block: &text},
		{Type: llm.EventContentStart, Index: 2, Block: &llm.ContentBlock{Type: llm.ContentToolCall, ToolCall: &llm.ToolCall{ID: "toolu_1", Name: "get_weather"}}},
		{Type: llm.EventToolCallDelta, Index: 2, Delta: ""},
		{Type: llm.EventToolCallDelta, Index: 2, Delta: `{"city":`},
		{Type: llm.EventToolCallDelta, Index: 2, Delta: `"SF"}`},
		{Type: llm.EventContentStop, Index: 2, Block: &call1},
		{Type: llm.EventContentStart, Index: 3, Block: &llm.ContentBlock{Type: llm.ContentToolCall, ToolCall: &llm.ToolCall{ID: "toolu_2", Name: "now"}}},
		{Type: llm.EventContentStop, Index: 3, Block: &call2},
		{Type: llm.EventDone, Response: &llm.Response{
			ID: "msg_1", Model: "claude-test",
			Content:    []llm.ContentBlock{thinking, text, call1, call2},
			StopReason: llm.StopToolUse, ProviderStopReason: "tool_use",
			Usage: llm.Usage{InputTokens: 15, OutputTokens: 42, CacheReadTokens: 5},
		}},
	}
	testutil.AssertEvents(t, events, want)
}

func TestStreamErrors(t *testing.T) {
	start := `{"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":1}}}`

	t.Run("http error before stream", func(t *testing.T) {
		srv, _ := testutil.Server(t, 401, "application/json",
			`{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`)
		_, err := newClient(srv).Stream(context.Background(), simpleRequest())
		var apiErr *llm.APIError
		if !errors.As(err, &apiErr) || apiErr.StatusCode != 401 || apiErr.Type != "authentication_error" || apiErr.Retryable() {
			t.Errorf("err = %v", err)
		}
	})

	cases := []struct {
		name  string
		body  string
		check func(error) bool
	}{
		{"error event", testutil.SSE(start, `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`),
			func(err error) bool {
				var e *llm.APIError
				return errors.As(err, &e) && e.Type == "overloaded_error" && e.StatusCode == 0 && e.Retryable()
			}},
		{"truncated", testutil.SSE(start, `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
			func(err error) bool { return errors.Is(err, llm.ErrIncompleteStream) }},
		{"malformed event", "event: message_start\ndata: {oops\n\n",
			func(err error) bool { return err != nil && strings.Contains(err.Error(), "decode") }},
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

func TestStreamContextCancel(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(testutil.SSE(`{"type":"message_start","message":{"id":"msg_1","model":"m"}}`)))
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	defer srv.Close()
	defer close(release)

	ctx, cancel := context.WithCancel(context.Background())
	s, err := newClient(srv).Stream(ctx, simpleRequest())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if !s.Next() || s.Event().Type != llm.EventStart {
		t.Fatalf("expected start event, err=%v", s.Err())
	}
	cancel()
	if s.Next() {
		t.Fatal("expected stream to stop after cancel")
	}
	if !errors.Is(s.Err(), context.Canceled) {
		t.Errorf("err = %v", s.Err())
	}
}

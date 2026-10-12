package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

func TestValidate(t *testing.T) {
	ok := &Request{Model: "m", Messages: []Message{UserText("hi")}}
	if err := ok.Validate(); err != nil {
		t.Fatalf("valid request: %v", err)
	}
	bad := []*Request{
		nil,
		{Messages: []Message{UserText("hi")}},
		{Model: "m"},
		{Model: "m", Messages: []Message{{Role: "system"}}},
		{Model: "m", Messages: []Message{UserText("hi")}, ToolChoice: &ToolChoice{Mode: ToolChoiceTool}},
	}
	for i, r := range bad {
		if err := r.Validate(); err == nil {
			t.Errorf("case %d: expected error", i)
		}
	}
}

func TestMessageHelpers(t *testing.T) {
	call := ToolCall{ID: "c1", Name: "f", Arguments: json.RawMessage(`{"a":1}`)}
	resp := &Response{Content: []ContentBlock{TextBlock("a"), ToolCallBlock(call), TextBlock("b")}}
	if resp.Text() != "ab" {
		t.Errorf("Text = %q", resp.Text())
	}
	if got := resp.ToolCalls(); !reflect.DeepEqual(got, []ToolCall{call}) {
		t.Errorf("ToolCalls = %v", got)
	}
	if m := resp.Message(); m.Role != RoleAssistant || len(m.Content) != 3 {
		t.Errorf("Message = %+v", m)
	}
	tr := ToolResults(ToolResult{ToolCallID: "c1", Content: "ok"})
	if tr.Role != RoleUser || tr.Content[0].ToolResult.ToolCallID != "c1" {
		t.Errorf("ToolResults = %+v", tr)
	}
}

func TestAPIError(t *testing.T) {
	tests := []struct {
		err       *APIError
		retryable bool
		text      string
	}{
		{&APIError{Provider: "p", StatusCode: 429, Type: "rate_limit_error", Message: "slow"}, true, "p: HTTP 429 (rate_limit_error): slow"},
		{&APIError{Provider: "p", StatusCode: 503}, true, "p: HTTP 503: Service Unavailable"},
		{&APIError{Provider: "p", StatusCode: 400, Code: "context_length_exceeded", Message: "long"}, false, "p: HTTP 400 (context_length_exceeded): long"},
		{&APIError{Provider: "p", Type: "overloaded_error", Message: "busy"}, true, "p: overloaded_error: busy"},
		{&APIError{Provider: "p", StatusCode: 401, Message: "no"}, false, "p: HTTP 401: no"},
	}
	for _, tt := range tests {
		if got := tt.err.Retryable(); got != tt.retryable {
			t.Errorf("%v: Retryable = %v", tt.err, got)
		}
		if got := tt.err.Error(); got != tt.text {
			t.Errorf("Error = %q, want %q", got, tt.text)
		}
		if got := IsRetryable(fmt.Errorf("wrapped: %w", tt.err)); got != tt.retryable {
			t.Errorf("IsRetryable(wrapped) = %v", got)
		}
	}
	if IsRetryable(errors.New("x")) {
		t.Error("plain error should not be retryable")
	}
}

type fakeStream struct {
	events []StreamEvent
	err    error
	i      int
	closed bool
}

func (f *fakeStream) Next() bool {
	if f.i >= len(f.events) {
		return false
	}
	f.i++
	return true
}
func (f *fakeStream) Event() StreamEvent { return f.events[f.i-1] }
func (f *fakeStream) Err() error         { return f.err }
func (f *fakeStream) Close() error       { f.closed = true; return nil }

func TestCollect(t *testing.T) {
	want := &Response{ID: "r"}
	s := &fakeStream{events: []StreamEvent{{Type: EventTextDelta}, {Type: EventDone, Response: want}}}
	got, err := Collect(s)
	if err != nil || got != want || !s.closed {
		t.Errorf("Collect = %v, %v (closed=%v)", got, err, s.closed)
	}

	if _, err := Collect(&fakeStream{events: []StreamEvent{{Type: EventTextDelta}}}); !errors.Is(err, ErrIncompleteStream) {
		t.Errorf("missing done: err = %v", err)
	}
	boom := errors.New("boom")
	if _, err := Collect(&fakeStream{err: boom}); !errors.Is(err, boom) {
		t.Errorf("stream error: err = %v", err)
	}
}

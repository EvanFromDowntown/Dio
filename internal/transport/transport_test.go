package transport

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/EvanFromDowntown/Dio/internal/sse"
	"github.com/EvanFromDowntown/Dio/llm"
)

func TestErrorFromBody(t *testing.T) {
	tests := []struct {
		name               string
		body               string
		typ, code, message string
	}{
		{"openai envelope", `{"error":{"message":"bad key","type":"invalid_request_error","code":"invalid_api_key"}}`,
			"invalid_request_error", "invalid_api_key", "bad key"},
		{"anthropic envelope", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`,
			"overloaded_error", "", "Overloaded"},
		{"bare stream error", `{"type":"error","code":"server_error","message":"boom"}`,
			"", "server_error", "boom"},
		{"numeric code", `{"error":{"message":"m","code":429}}`, "", "429", "m"},
		{"plain text", "upstream unavailable\n", "", "", "upstream unavailable"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := ErrorFromBody("p", []byte(tt.body))
			if e.Type != tt.typ || e.Code != tt.code || e.Message != tt.message {
				t.Errorf("got type=%q code=%q message=%q", e.Type, e.Code, e.Message)
			}
			if string(e.Body) != tt.body {
				t.Errorf("body not preserved")
			}
		})
	}
}

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	if got := ParseRetryAfter("3", now); got != 3*time.Second {
		t.Errorf("seconds: got %v", got)
	}
	if got := ParseRetryAfter(now.Add(10*time.Second).Format(http.TimeFormat), now); got != 10*time.Second {
		t.Errorf("date: got %v", got)
	}
	if got := ParseRetryAfter("garbage", now); got != 0 {
		t.Errorf("garbage: got %v", got)
	}
}

func TestMarshalWithExtra(t *testing.T) {
	body, err := MarshalWithExtra(struct {
		A int `json:"a"`
		B int `json:"b"`
	}{1, 2}, map[string]any{"b": 3, "store": false})
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	want := map[string]any{"a": 1.0, "b": 3.0, "store": false}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

type closeTracker struct {
	io.Reader
	closed bool
}

func (c *closeTracker) Close() error { c.closed = true; return nil }

func TestStreamLifecycle(t *testing.T) {
	decode := func(ev sse.Event) ([]llm.StreamEvent, bool, error) {
		switch ev.Name {
		case "two":
			return []llm.StreamEvent{{Type: llm.EventTextDelta, Delta: "a"}, {Type: llm.EventTextDelta, Delta: "b"}}, false, nil
		case "done":
			return []llm.StreamEvent{{Type: llm.EventDone, Response: &llm.Response{ID: "r"}}}, true, nil
		case "fail":
			return nil, false, errors.New("fail")
		}
		return nil, false, nil
	}
	run := func(input string) ([]llm.StreamEvent, error, bool) {
		body := &closeTracker{Reader: strings.NewReader(input)}
		s := NewStream(body, decode)
		var got []llm.StreamEvent
		for s.Next() {
			got = append(got, s.Event())
		}
		return got, s.Err(), body.closed
	}

	got, err, closed := run("event: ignored\ndata: x\n\nevent: two\ndata: x\n\nevent: done\ndata: x\n\nevent: two\ndata: x\n\n")
	if err != nil || len(got) != 3 || got[2].Type != llm.EventDone || !closed {
		t.Errorf("complete stream: events=%v err=%v closed=%v", got, err, closed)
	}

	_, err, closed = run("event: two\ndata: x\n\n")
	if !errors.Is(err, llm.ErrIncompleteStream) || !closed {
		t.Errorf("truncated stream: err=%v closed=%v", err, closed)
	}

	finishOnEOF := NewStreamWithEOF(io.NopCloser(strings.NewReader("event: two\ndata: x\n\n")), decode,
		func() ([]llm.StreamEvent, bool, error) {
			return []llm.StreamEvent{{Type: llm.EventDone}}, true, nil
		})
	n := 0
	for finishOnEOF.Next() {
		n++
	}
	if finishOnEOF.Err() != nil || n != 3 {
		t.Errorf("EOF completion: events=%d err=%v", n, finishOnEOF.Err())
	}

	got, err, _ = run("event: two\ndata: x\n\nevent: fail\ndata: x\n\n")
	if err == nil || err.Error() != "fail" || len(got) != 2 {
		t.Errorf("failing stream: events=%v err=%v", got, err)
	}
}

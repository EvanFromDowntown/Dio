// Package testutil provides httptest helpers shared by adapter tests.
package testutil

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/EvanFromDowntown/Dio/llm"
)

// Captured is the last request received by a Server.
type Captured struct {
	Method, Path string
	Header       http.Header
	Body         []byte
}

// Server serves a fixed status, content type and body. header holds
// alternating response header keys and values.
func Server(t *testing.T, status int, contentType, body string, header ...string) (*httptest.Server, *Captured) {
	t.Helper()
	got := &Captured{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.Method, got.Path, got.Header = r.Method, r.URL.Path, r.Header.Clone()
		got.Body, _ = io.ReadAll(r.Body)
		for i := 0; i+1 < len(header); i += 2 {
			w.Header().Set(header[i], header[i+1])
		}
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(status)
		io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv, got
}

// AssertJSON compares two JSON documents structurally.
func AssertJSON(t *testing.T, got []byte, want string) {
	t.Helper()
	var g, w any
	if err := json.Unmarshal(got, &g); err != nil {
		t.Fatalf("invalid JSON %s: %v", got, err)
	}
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("invalid want JSON: %v", err)
	}
	if !reflect.DeepEqual(g, w) {
		t.Errorf("JSON mismatch\ngot:  %s\nwant: %s", Pretty(g), Pretty(w))
	}
}

func Pretty(v any) string {
	b, _ := json.MarshalIndent(v, "", "  ")
	return string(b)
}

// SSE renders JSON payloads as an event stream, naming each event after the
// payload's "type" field.
func SSE(payloads ...string) string {
	var b strings.Builder
	for _, p := range payloads {
		var head struct{ Type string }
		var compact bytes.Buffer
		if err := json.Unmarshal([]byte(p), &head); err != nil {
			panic(err)
		}
		if err := json.Compact(&compact, []byte(p)); err != nil {
			panic(err)
		}
		b.WriteString("event: " + head.Type + "\ndata: " + compact.String() + "\n\n")
	}
	return b.String()
}

// Drain reads every event from s and fails the test on a stream error.
func Drain(t *testing.T, s llm.Stream) []llm.StreamEvent {
	t.Helper()
	defer s.Close()
	var events []llm.StreamEvent
	for s.Next() {
		events = append(events, s.Event())
	}
	if err := s.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}
	return events
}

// AssertEvents compares event sequences and prints both as JSON on mismatch.
func AssertEvents(t *testing.T, got, want []llm.StreamEvent) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Errorf("events mismatch\ngot:  %s\nwant: %s", Pretty(got), Pretty(want))
	}
}

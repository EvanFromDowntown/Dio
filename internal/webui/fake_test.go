package webui

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/session"
)

type step func(ctx context.Context, req *llm.Request) (*llm.Response, error)

type fakeClient struct {
	mu       sync.Mutex
	steps    []step
	requests []*llm.Request
}

func (f *fakeClient) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
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

func (f *fakeClient) Stream(ctx context.Context, req *llm.Request) (llm.Stream, error) {
	resp, err := f.Generate(ctx, req)
	if err != nil {
		return nil, err
	}
	events := []llm.StreamEvent{{Type: llm.EventStart, Response: &llm.Response{}}}
	for i, b := range resp.Content {
		switch b.Type {
		case llm.ContentText:
			runes := []rune(b.Text)
			half := len(runes) / 2
			events = append(events,
				llm.StreamEvent{Type: llm.EventTextDelta, Index: i, Delta: string(runes[:half])},
				llm.StreamEvent{Type: llm.EventTextDelta, Index: i, Delta: string(runes[half:])})
		case llm.ContentReasoning:
			events = append(events, llm.StreamEvent{Type: llm.EventReasoningDelta, Index: i, Delta: b.Reasoning.Text})
		}
	}
	events = append(events, llm.StreamEvent{Type: llm.EventDone, Response: resp})
	return &sliceStream{events: events}, nil
}

func (f *fakeClient) calls() []*llm.Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.requests)
}

type sliceStream struct {
	events []llm.StreamEvent
	i      int
}

func (s *sliceStream) Next() bool             { s.i++; return s.i <= len(s.events) }
func (s *sliceStream) Event() llm.StreamEvent { return s.events[s.i-1] }
func (s *sliceStream) Err() error             { return nil }
func (s *sliceStream) Close() error           { return nil }

func answer(text string) step {
	return func(context.Context, *llm.Request) (*llm.Response, error) {
		return &llm.Response{
			Content: []llm.ContentBlock{
				{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: "fake", Text: "想一想"}},
				llm.TextBlock(text),
			},
			StopReason: llm.StopEndTurn, Usage: llm.Usage{InputTokens: 100, OutputTokens: 10},
		}, nil
	}
}

func readFile(path string) step {
	return func(context.Context, *llm.Request) (*llm.Response, error) {
		return &llm.Response{
			Content: []llm.ContentBlock{
				llm.TextBlock("让我看看。"),
				llm.ToolCallBlock(llm.ToolCall{ID: "c-" + path, Name: "read_file", Arguments: json.RawMessage(`{"path": "` + path + `"}`)}),
			},
			StopReason: llm.StopToolUse, Usage: llm.Usage{InputTokens: 80, OutputTokens: 5},
		}, nil
	}
}

type env struct {
	t       *testing.T
	srv     *Server
	http    *httptest.Server
	client  *fakeClient
	store   *session.MemoryStore
	root    string
	factory struct {
		mu    sync.Mutex
		calls []string
	}
}

const testKey = "sk-test-secret-value"

func newEnv(t *testing.T, apiKey string, steps ...step) *env {
	t.Helper()
	e := &env{t: t, client: &fakeClient{steps: steps}, store: session.NewMemoryStore(), root: t.TempDir()}
	os.WriteFile(filepath.Join(e.root, "notes.txt"), []byte("Project codename: Falcon-7\nLaunch: 2026-11-03\n"), 0o644)
	factory := func(provider, baseURL, key string) (llm.Client, error) {
		e.factory.mu.Lock()
		e.factory.calls = append(e.factory.calls, provider+"|"+baseURL+"|"+key)
		e.factory.mu.Unlock()
		return e.client, nil
	}
	settings := Settings{Provider: "chatcompletions", Model: "fake-model", BaseURL: "https://gw/v1", Root: e.root, MaxTurns: 5, MaxTokens: 1000}
	var err error
	e.srv, err = New(e.store, settings, apiKey, WithClientFactory(factory))
	if err != nil {
		t.Fatal(err)
	}
	e.http = httptest.NewServer(e.srv.Handler())
	t.Cleanup(e.http.Close)
	return e
}

// do sends a JSON request and decodes a JSON response into out (if non-nil).
func (e *env) do(method, path string, body any, out any) int {
	e.t.Helper()
	var r io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		r = bytes.NewReader(b)
	}
	req, _ := http.NewRequest(method, e.http.URL+path, r)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := e.http.Client().Do(req)
	if err != nil {
		e.t.Fatal(err)
	}
	defer resp.Body.Close()
	if out != nil {
		if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
			e.t.Fatalf("%s %s: decode: %v", method, path, err)
		}
	}
	return resp.StatusCode
}

func (e *env) createSession() SessionSummary {
	e.t.Helper()
	var s SessionSummary
	if code := e.do("POST", "/api/sessions", map[string]any{}, &s); code != http.StatusCreated {
		e.t.Fatalf("create session: %d", code)
	}
	return s
}

type sseEvent struct {
	Name string
	Data json.RawMessage
}

func parseSSE(t *testing.T, r io.Reader) []sseEvent {
	t.Helper()
	var events []sseEvent
	sc := bufio.NewScanner(r)
	var cur sseEvent
	for sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			cur.Name = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			cur.Data = json.RawMessage(strings.TrimPrefix(line, "data: "))
		case line == "" && cur.Name != "":
			events = append(events, cur)
			cur = sseEvent{}
		}
	}
	return events
}

// chat posts a message and returns the SSE events.
func (e *env) chat(id, text string) []sseEvent {
	e.t.Helper()
	b, _ := json.Marshal(map[string]string{"text": text})
	resp, err := e.http.Client().Post(e.http.URL+"/api/sessions/"+id+"/messages", "application/json", bytes.NewReader(b))
	if err != nil {
		e.t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		body, _ := io.ReadAll(resp.Body)
		e.t.Fatalf("chat: %d %s", resp.StatusCode, body)
	}
	return parseSSE(e.t, resp.Body)
}

func doneOf(t *testing.T, events []sseEvent) doneEvent {
	t.Helper()
	if len(events) == 0 || events[len(events)-1].Name != "done" {
		t.Fatalf("stream did not end with done: %+v", events)
	}
	var d doneEvent
	json.Unmarshal(events[len(events)-1].Data, &d)
	return d
}

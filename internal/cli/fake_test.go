package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/session"
)

// step scripts one model call.
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

// Stream replays the response as deltas, splitting text in two chunks.
func (f *fakeClient) Stream(ctx context.Context, req *llm.Request) (llm.Stream, error) {
	resp, err := f.Generate(ctx, req)
	if err != nil {
		return nil, err
	}
	events := []llm.StreamEvent{{Type: llm.EventStart, Response: &llm.Response{}}}
	for i, b := range resp.Content {
		b := b
		events = append(events, llm.StreamEvent{Type: llm.EventContentStart, Index: i, Block: &b})
		switch b.Type {
		case llm.ContentText:
			half := len(b.Text) / 2
			events = append(events,
				llm.StreamEvent{Type: llm.EventTextDelta, Index: i, Delta: b.Text[:half]},
				llm.StreamEvent{Type: llm.EventTextDelta, Index: i, Delta: b.Text[half:]})
		case llm.ContentReasoning:
			events = append(events, llm.StreamEvent{Type: llm.EventReasoningDelta, Index: i, Delta: b.Reasoning.Text})
		case llm.ContentToolCall:
			events = append(events, llm.StreamEvent{Type: llm.EventToolCallDelta, Index: i, Delta: string(b.ToolCall.Arguments)})
		}
		events = append(events, llm.StreamEvent{Type: llm.EventContentStop, Index: i, Block: &b})
	}
	events = append(events, llm.StreamEvent{Type: llm.EventDone, Response: resp})
	return &sliceStream{events: events}, nil
}

func (f *fakeClient) lastRequest() *llm.Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.requests[len(f.requests)-1]
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
				{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: "fake", Text: "secret thoughts"}},
				llm.TextBlock(text),
			},
			StopReason: llm.StopEndTurn, Usage: llm.Usage{InputTokens: 100, OutputTokens: 10, CacheReadTokens: 50},
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

func noEnv(string) string { return "" }

type harness struct {
	app    *App
	client *fakeClient
	store  *session.MemoryStore
	out    *bytes.Buffer
	info   *bytes.Buffer
	opts   *Options
}

// newHarness builds an App over a temp root containing notes.txt. A nil
// store gets a fresh MemoryStore.
func newHarness(t *testing.T, store *session.MemoryStore, args []string, steps ...step) *harness {
	t.Helper()
	root := t.TempDir()
	os.WriteFile(filepath.Join(root, "notes.txt"), []byte("Project codename: Falcon-7\nLaunch: 2026-11-03\n"), 0o644)
	base := []string{"-root", root}
	if !slices.Contains(args, "-session") && !slices.Contains(args, "-model") {
		base = append(base, "-model", "fake-model")
	}
	opts, err := Parse(append(base, args...), noEnv, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if store == nil {
		store = session.NewMemoryStore()
	}
	if err := ApplySessionDefaults(context.Background(), opts, store); err != nil {
		t.Fatal(err)
	}
	h := &harness{client: &fakeClient{steps: steps}, store: store, out: &bytes.Buffer{}, info: &bytes.Buffer{}, opts: opts}
	h.app, err = NewApp(context.Background(), opts, h.client, store, h.out, h.info)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { h.app.Close() })
	return h
}

// interactive routes status lines to the answer stream, as dio does on a TTY.
func (h *harness) interactive() *harness {
	h.app.Info = h.app.Out
	return h
}

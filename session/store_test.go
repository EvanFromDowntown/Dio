package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/EvanFromDowntown/Dio/llm"
)

type clock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *clock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(time.Second)
	return c.t
}

func newClock() *clock { return &clock{t: time.Date(2026, 10, 12, 9, 0, 0, 0, time.UTC)} }

// stores returns a fresh instance of every Store implementation.
func stores(t *testing.T) map[string]Store {
	t.Helper()
	fs, err := NewFileStore(filepath.Join(t.TempDir(), "sessions"), WithClock(newClock().now))
	if err != nil {
		t.Fatal(err)
	}
	mem := NewMemoryStore()
	mem.now = newClock().now
	return map[string]Store{"file": fs, "memory": mem}
}

// richHistory uses every content block type, including provider-specific
// reasoning and non-JSON tool arguments.
func richHistory() []llm.Message {
	return []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{
			llm.TextBlock("看看这张图 and the weather"),
			llm.ImageURLBlock("https://x/a.png"),
			llm.ImageDataBlock("image/png", "AAAA"),
		}},
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: "anthropic", Text: "thinking", Signature: "sig=="}},
			{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: "anthropic", Encrypted: "redacted"}},
			{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: "openai", ID: "rs_1", Text: "summary", Encrypted: "enc"}},
			{Type: llm.ContentReasoning, Reasoning: &llm.Reasoning{Provider: "chatcompletions", Text: "raw cot"}},
			llm.TextBlock("Checking."),
			llm.ToolCallBlock(llm.ToolCall{ID: "c1", Name: "get_weather", Arguments: json.RawMessage(`{"city": "Beijing"}`)}),
			llm.ToolCallBlock(llm.ToolCall{ID: "c2", Name: "broken", Arguments: json.RawMessage(`{"city": `)}),
		}},
		llm.ToolResults(
			llm.ToolResult{ToolCallID: "c1", Content: "Sunny\n\"25C\""},
			llm.ToolResult{ToolCallID: "c2", Content: "invalid arguments", IsError: true},
		),
		llm.AssistantText("Sunny, 25C."),
	}
}

func TestStoreLifecycle(t *testing.T) {
	ctx := context.Background()
	for name, s := range stores(t) {
		t.Run(name, func(t *testing.T) {
			created, err := s.Create(ctx, Info{Provider: "chatcompletions", Model: "deepseek-flash", Metadata: map[string]string{"cwd": "/tmp"}})
			if err != nil {
				t.Fatal(err)
			}
			if ValidateID(created.ID) != nil || created.CreatedAt.IsZero() {
				t.Fatalf("created = %+v", created)
			}
			if _, err := s.Create(ctx, Info{ID: created.ID}); !errors.Is(err, ErrExists) {
				t.Errorf("duplicate create: %v", err)
			}
			second, err := s.Create(ctx, Info{ID: "custom-id"})
			if err != nil {
				t.Fatal(err)
			}

			history := richHistory()
			if err := s.AppendMessages(ctx, created.ID, history[:2]...); err != nil {
				t.Fatal(err)
			}
			if err := s.AddUsage(ctx, created.ID, llm.Usage{InputTokens: 10, OutputTokens: 2, CacheReadTokens: 4}); err != nil {
				t.Fatal(err)
			}
			if err := s.AppendMessages(ctx, created.ID, history[2:]...); err != nil {
				t.Fatal(err)
			}
			if err := s.AddUsage(ctx, created.ID, llm.Usage{InputTokens: 5, OutputTokens: 1, ReasoningTokens: 1}); err != nil {
				t.Fatal(err)
			}
			if err := s.SetTitle(ctx, created.ID, "weather"); err != nil {
				t.Fatal(err)
			}

			got, err := s.Get(ctx, created.ID)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got.Messages, history) {
				t.Errorf("messages did not round-trip:\ngot  %+v\nwant %+v", got.Messages, history)
			}
			wantUsage := llm.Usage{InputTokens: 15, OutputTokens: 3, CacheReadTokens: 4, ReasoningTokens: 1}
			if got.Usage != wantUsage || got.Title != "weather" || got.MessageCount != 4 ||
				got.Provider != "chatcompletions" || got.Model != "deepseek-flash" || got.Metadata["cwd"] != "/tmp" ||
				!got.CreatedAt.Equal(created.CreatedAt) || !got.UpdatedAt.After(got.CreatedAt) {
				t.Errorf("info = %+v", got.Info)
			}

			list, err := s.List(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if len(list) != 2 || list[0].ID != created.ID || list[1].ID != second.ID {
				t.Errorf("list = %+v", list)
			}

			if err := s.Delete(ctx, created.ID); err != nil {
				t.Fatal(err)
			}
			for _, err := range []error{
				func() error { _, err := s.Get(ctx, created.ID); return err }(),
				s.AppendMessages(ctx, created.ID, llm.UserText("x")),
				s.AddUsage(ctx, "missing", llm.Usage{}),
				s.Delete(ctx, created.ID),
			} {
				if !errors.Is(err, ErrNotFound) {
					t.Errorf("expected ErrNotFound, got %v", err)
				}
			}
		})
	}
}

func TestInvalidIDs(t *testing.T) {
	ctx := context.Background()
	for name, s := range stores(t) {
		for _, id := range []string{"../escape", "a/b", ".hidden", "with space", strings.Repeat("x", 200)} {
			if _, err := s.Create(ctx, Info{ID: id}); !errors.Is(err, ErrInvalidID) {
				t.Errorf("%s: Create(%q) err = %v", name, id, err)
			}
		}
	}
	fs := stores(t)["file"]
	for _, err := range []error{
		func() error { _, err := fs.Get(ctx, "../x"); return err }(),
		fs.AppendMessages(ctx, "../x", llm.UserText("x")),
		fs.Delete(ctx, "../x"),
	} {
		if !errors.Is(err, ErrInvalidID) {
			t.Errorf("file store accepted traversal id: %v", err)
		}
	}
}

func TestFileStoreTornTail(t *testing.T) {
	ctx := context.Background()
	s, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	sess, err := s.Create(ctx, Info{ID: "torn"})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.AppendMessages(ctx, "torn", llm.UserText("first")); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(s.Dir(), "torn.jsonl")
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString(`{"type":"message","at":"2026-10-12T00:00:00Z","message":{"role":"assis`)
	f.Close()

	got, err := s.Get(ctx, sess.ID)
	if err != nil {
		t.Fatalf("torn tail should be ignored: %v", err)
	}
	if len(got.Messages) != 1 || got.Messages[0].Text() != "first" {
		t.Errorf("messages = %+v", got.Messages)
	}

	if err := s.AppendMessages(ctx, "torn", llm.AssistantText("second")); err != nil {
		t.Fatal(err)
	}
	got, err = s.Get(ctx, "torn")
	if err != nil {
		t.Fatalf("append after torn tail: %v", err)
	}
	if len(got.Messages) != 2 || got.Messages[1].Text() != "second" {
		t.Errorf("messages = %+v", got.Messages)
	}
	data, _ := os.ReadFile(path)
	if strings.Contains(string(data), `"role":"assis`+"\n") || strings.Count(string(data), "\n") != 3 {
		t.Errorf("torn line not removed:\n%s", data)
	}
}

func TestFileStoreCorruption(t *testing.T) {
	ctx := context.Background()
	write := func(t *testing.T, content string) (*FileStore, string) {
		t.Helper()
		s, err := NewFileStore(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		os.WriteFile(filepath.Join(s.Dir(), "bad.jsonl"), []byte(content), 0o600)
		return s, "bad"
	}
	header := `{"type":"session","at":"2026-10-12T00:00:00Z","session":{"version":1,"id":"bad"}}` + "\n"
	msg := `{"type":"message","at":"2026-10-12T00:00:01Z","message":{"role":"user","content":[{"type":"text","text":"hi"}]}}` + "\n"

	cases := map[string]string{
		"corrupt middle line":            header + "not json\n" + msg,
		"complete but invalid last line": header + msg + "{oops}\n",
		"missing header":                 msg,
		"torn header only":               `{"type":"sess`,
	}
	for name, content := range cases {
		t.Run(name, func(t *testing.T) {
			s, id := write(t, content)
			if _, err := s.Get(ctx, id); err == nil {
				t.Error("expected error")
			}
			if list, _ := s.List(ctx); len(list) != 0 {
				t.Errorf("corrupt session listed: %+v", list)
			}
		})
	}

	s, id := write(t, header+msg+`{"type":"usage","at":"2026-10-12T00:00:02Z","usage":{"input_tokens":1},"x":`)
	got, err := s.Get(ctx, id)
	if err != nil || len(got.Messages) != 1 {
		t.Errorf("torn usage tail: %v %+v", err, got)
	}

	s, id = write(t, header+`{"type":"future_record","at":"2026-10-12T00:00:03Z"}`+"\n"+msg)
	if got, err := s.Get(ctx, id); err != nil || len(got.Messages) != 1 {
		t.Errorf("unknown record type should be skipped: %v", err)
	}
}

func TestConcurrentAppends(t *testing.T) {
	ctx := context.Background()
	for name, s := range stores(t) {
		t.Run(name, func(t *testing.T) {
			ids := []string{"a", "b"}
			for _, id := range ids {
				if _, err := s.Create(ctx, Info{ID: id}); err != nil {
					t.Fatal(err)
				}
			}
			const workers, perWorker = 8, 25
			var wg sync.WaitGroup
			for w := range workers {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for i := range perWorker {
						id := ids[(w+i)%2]
						if err := s.AppendMessages(ctx, id, llm.UserText(fmt.Sprintf("%d-%d", w, i))); err != nil {
							t.Error(err)
						}
						if err := s.AddUsage(ctx, id, llm.Usage{OutputTokens: 1}); err != nil {
							t.Error(err)
						}
						if _, err := s.Get(ctx, id); err != nil {
							t.Error(err)
						}
						if _, err := s.List(ctx); err != nil {
							t.Error(err)
						}
					}
				}()
			}
			wg.Wait()
			total, seen := 0, map[string]bool{}
			for _, id := range ids {
				got, err := s.Get(ctx, id)
				if err != nil {
					t.Fatal(err)
				}
				total += len(got.Messages)
				if got.Usage.OutputTokens != len(got.Messages) {
					t.Errorf("%s: usage %d for %d messages", id, got.Usage.OutputTokens, len(got.Messages))
				}
				for _, m := range got.Messages {
					seen[m.Text()] = true
				}
			}
			if total != workers*perWorker || len(seen) != total {
				t.Errorf("total=%d unique=%d, want %d", total, len(seen), workers*perWorker)
			}
		})
	}
}

func TestPendingToolCallsAndTitle(t *testing.T) {
	h := richHistory()
	if got := PendingToolCalls(h); got != nil {
		t.Errorf("complete history has pending calls: %v", got)
	}
	if got := PendingToolCalls(h[:2]); len(got) != 2 || got[0].ID != "c1" {
		t.Errorf("pending = %v", got)
	}
	if got := Title(h); got != "看看这张图 and the weather" {
		t.Errorf("title = %q", got)
	}
	long := strings.Repeat("长", 70)
	if got := Title([]llm.Message{llm.UserText(long)}); len([]rune(got)) != 61 {
		t.Errorf("long title = %q", got)
	}
}

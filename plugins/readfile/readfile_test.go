package readfile

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/EvanFromDowntown/Dio/agent"
	"github.com/EvanFromDowntown/Dio/llm"
)

type host struct{ logger *slog.Logger }

func (h host) Logger() *slog.Logger { return h.logger }

func setup(t *testing.T) (*Plugin, *bytes.Buffer) {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "hello.txt"), []byte("hello world"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(t.TempDir(), "secret.txt"), []byte("secret"), 0o644); err != nil {
		t.Fatal(err)
	}
	var logs bytes.Buffer
	p := New(dir)
	p.MaxBytes = 5
	if err := p.Init(context.Background(), host{slog.New(slog.NewTextHandler(&logs, nil))}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { p.Close() })
	return p, &logs
}

func TestReadFile(t *testing.T) {
	p, _ := setup(t)
	read := p.Tools()[0]
	if read.Name() != "read_file" || !json.Valid(read.Schema()) {
		t.Fatalf("unexpected tool %q", read.Name())
	}
	ctx := context.Background()

	out, err := read.Execute(ctx, json.RawMessage(`{"path":"hello.txt"}`))
	if err != nil || out != "hello\n[truncated: showing first 5 of 11 bytes]" {
		t.Errorf("read = %q, %v", out, err)
	}
	for args, want := range map[string]string{
		`{"path":"../secret.txt"}`: "escapes",
		`{"path":"/etc/hosts"}`:    "escapes",
		`{"path":"sub"}`:           "is a directory",
		`{"path":"nope.txt"}`:      "no such file",
		`{}`:                       "path is required",
		`not json`:                 "invalid arguments",
	} {
		if _, err := read.Execute(ctx, json.RawMessage(args)); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: err = %v, want %q", args, err, want)
		}
	}
}

func TestInitFailsForMissingRoot(t *testing.T) {
	if err := New(filepath.Join(t.TempDir(), "missing")).Init(context.Background(), host{slog.Default()}); err == nil {
		t.Error("expected error")
	}
}

type scripted struct{ responses []*llm.Response }

func (s *scripted) Generate(context.Context, *llm.Request) (*llm.Response, error) {
	r := s.responses[0]
	s.responses = s.responses[1:]
	return r, nil
}

func (s *scripted) Stream(context.Context, *llm.Request) (llm.Stream, error) { panic("unused") }

func TestPluginInAgent(t *testing.T) {
	setupPlugin, _ := setup(t)
	p := New(setupPlugin.Root)
	defer p.Close()
	var logs bytes.Buffer
	client := &scripted{responses: []*llm.Response{
		{Content: []llm.ContentBlock{llm.ToolCallBlock(llm.ToolCall{ID: "c1", Name: "read_file", Arguments: json.RawMessage(`{"path":"hello.txt"}`)})}},
		{Content: []llm.ContentBlock{llm.TextBlock("It says hello world.")}},
	}}
	a, err := agent.New(context.Background(), agent.Config{
		Client: client, Model: "m", Plugins: []agent.Plugin{p},
		Logger: slog.New(slog.NewTextHandler(&logs, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := a.Run(context.Background(), []llm.Message{llm.UserText("what does hello.txt say?")})
	if err != nil {
		t.Fatal(err)
	}
	if got := res.Messages[2].Content[0].ToolResult.Content; got != "hello world" {
		t.Errorf("tool result = %q", got)
	}
	for _, want := range []string{"msg=\"tool call\"", "msg=\"tool result\"", "plugin=readfile", "reason=completed turns=2"} {
		if !strings.Contains(logs.String(), want) {
			t.Errorf("log missing %q:\n%s", want, logs.String())
		}
	}
}

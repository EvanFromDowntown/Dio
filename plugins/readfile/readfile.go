// Package readfile is a minimal example plugin: a read-only read_file tool
// confined to a root directory, plus hooks that log tool calls and runs.
package readfile

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"

	"github.com/EvanFromDowntown/Dio/agent"
	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/tool"
)

const DefaultMaxBytes = 64 << 10

type Plugin struct {
	// Root is the directory files are read from; paths cannot escape it.
	Root string
	// MaxBytes caps how much of a file is returned. Zero means
	// DefaultMaxBytes.
	MaxBytes int64

	root   *os.Root
	logger *slog.Logger
}

var (
	_ agent.ToolProvider   = (*Plugin)(nil)
	_ agent.PromptProvider = (*Plugin)(nil)
	_ agent.ToolHook       = (*Plugin)(nil)
	_ agent.FinishHook     = (*Plugin)(nil)
)

func New(root string) *Plugin { return &Plugin{Root: root} }

func (p *Plugin) Name() string { return "readfile" }

func (p *Plugin) Init(_ context.Context, host agent.Host) error {
	root, err := os.OpenRoot(p.Root)
	if err != nil {
		return err
	}
	if p.MaxBytes <= 0 {
		p.MaxBytes = DefaultMaxBytes
	}
	if p.root != nil {
		p.root.Close()
	}
	p.root, p.logger = root, host.Logger()
	return nil
}

// Close releases the root directory handle.
func (p *Plugin) Close() error {
	if p.root == nil {
		return nil
	}
	return p.root.Close()
}

func (p *Plugin) Tools() []tool.Tool {
	return []tool.Tool{&tool.Func{
		ToolName:        "read_file",
		ToolDescription: "Read a UTF-8 text file. The path is relative to the workspace root.",
		InputSchema: json.RawMessage(`{
			"type": "object",
			"properties": {"path": {"type": "string", "description": "File path relative to the workspace root"}},
			"required": ["path"]
		}`),
		Fn: p.readFile,
	}}
}

func (p *Plugin) SystemPrompt(context.Context) (string, error) {
	return "You can inspect files in the workspace with the read_file tool. It is read-only.", nil
}

func (p *Plugin) readFile(_ context.Context, raw json.RawMessage) (string, error) {
	var args struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(raw, &args); err != nil {
		return "", fmt.Errorf("invalid arguments: %w", err)
	}
	if args.Path == "" {
		return "", errors.New("path is required")
	}
	f, err := p.root.Open(args.Path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		return "", fmt.Errorf("%s is a directory", args.Path)
	}
	data, err := io.ReadAll(io.LimitReader(f, p.MaxBytes+1))
	if err != nil {
		return "", err
	}
	if int64(len(data)) > p.MaxBytes {
		return fmt.Sprintf("%s\n[truncated: showing first %d of %d bytes]", data[:p.MaxBytes], p.MaxBytes, info.Size()), nil
	}
	return string(data), nil
}

func (p *Plugin) BeforeTool(_ context.Context, call *llm.ToolCall) (agent.Decision, error) {
	p.logger.Info("tool call", "tool", call.Name, "id", call.ID, "args", string(call.Arguments))
	return agent.Allow(), nil
}

func (p *Plugin) AfterTool(_ context.Context, call llm.ToolCall, result *llm.ToolResult) error {
	p.logger.Info("tool result", "tool", call.Name, "id", call.ID, "is_error", result.IsError, "bytes", len(result.Content))
	return nil
}

func (p *Plugin) OnFinish(_ context.Context, res *agent.Result) error {
	p.logger.Info("run finished", "reason", res.FinishReason, "turns", res.Turns,
		"input_tokens", res.Usage.InputTokens, "output_tokens", res.Usage.OutputTokens)
	return nil
}

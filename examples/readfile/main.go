// Command readfile runs a one-shot agent that can read files under -root.
//
//	OPENAI_API_KEY=... go run ./examples/readfile -model gpt-5 "Summarize README.md"
//	ANTHROPIC_API_KEY=... go run ./examples/readfile -provider anthropic -model claude-sonnet-4-5 "Summarize README.md"
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"

	"github.com/EvanFromDowntown/Dio/agent"
	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/llm/anthropic"
	"github.com/EvanFromDowntown/Dio/llm/openai"
	"github.com/EvanFromDowntown/Dio/plugins/readfile"
)

func main() {
	provider := flag.String("provider", "openai", "openai or anthropic")
	model := flag.String("model", "", "model name (required)")
	baseURL := flag.String("base-url", "", "override the provider base URL")
	root := flag.String("root", ".", "directory read_file may access")
	maxTurns := flag.Int("max-turns", 10, "maximum model calls")
	flag.Parse()
	if *model == "" || flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: readfile -model MODEL [flags] PROMPT")
		os.Exit(2)
	}
	if err := run(*provider, *model, *baseURL, *root, *maxTurns, strings.Join(flag.Args(), " ")); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(provider, model, baseURL, root string, maxTurns int, prompt string) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	var client llm.Client
	switch provider {
	case "openai":
		opts := []openai.Option{openai.WithAPIKey(os.Getenv("OPENAI_API_KEY"))}
		if baseURL != "" {
			opts = append(opts, openai.WithBaseURL(baseURL))
		}
		client = openai.New(opts...)
	case "anthropic":
		opts := []anthropic.Option{anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY"))}
		if baseURL != "" {
			opts = append(opts, anthropic.WithBaseURL(baseURL))
		}
		client = anthropic.New(opts...)
	default:
		return fmt.Errorf("unknown provider %q", provider)
	}

	files := readfile.New(root)
	defer files.Close()
	a, err := agent.New(ctx, agent.Config{
		Client:   client,
		Model:    model,
		System:   "You are a concise assistant.",
		MaxTurns: maxTurns,
		Plugins:  []agent.Plugin{files},
		Logger:   slog.New(slog.NewTextHandler(os.Stderr, nil)),
	})
	if err != nil {
		return err
	}

	for ev, err := range a.Stream(ctx, []llm.Message{llm.UserText(prompt)}) {
		if err != nil {
			return err
		}
		if ev.Type == agent.EventModel && ev.LLM.Type == llm.EventTextDelta {
			fmt.Print(ev.LLM.Delta)
		}
		if ev.Type == agent.EventModelResponse && ev.Response.Text() != "" {
			fmt.Println()
		}
	}
	return nil
}

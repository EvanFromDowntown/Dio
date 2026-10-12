// Command readfile runs a one-shot agent that can read files under -root.
//
//	OPENAI_API_KEY=... go run ./examples/readfile -model gpt-5 "Summarize README.md"
//	ANTHROPIC_API_KEY=... go run ./examples/readfile -provider anthropic -model claude-sonnet-4-5 "Summarize README.md"
//	DIO_API_KEY=... go run ./examples/readfile -provider chatcompletions -api-key-env DIO_API_KEY \
//		-base-url https://gateway.example.com/v1 -model deepseek-flash "Summarize README.md"
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
	"github.com/EvanFromDowntown/Dio/llm/chatcompletions"
	"github.com/EvanFromDowntown/Dio/llm/openai"
	"github.com/EvanFromDowntown/Dio/plugins/readfile"
)

var defaultKeyEnv = map[string]string{
	"openai":          "OPENAI_API_KEY",
	"chatcompletions": "OPENAI_API_KEY",
	"anthropic":       "ANTHROPIC_API_KEY",
}

type options struct {
	provider, model, baseURL, keyEnv, root, effort string
	maxTurns, maxTokens                            int
	stream                                         bool
}

func main() {
	var o options
	flag.StringVar(&o.provider, "provider", "openai", "openai (Responses), chatcompletions, or anthropic (Messages)")
	flag.StringVar(&o.model, "model", "", "model name (required)")
	flag.StringVar(&o.baseURL, "base-url", "", "override the provider base URL")
	flag.StringVar(&o.keyEnv, "api-key-env", "", "environment variable holding the API key (default depends on -provider)")
	flag.StringVar(&o.root, "root", ".", "directory read_file may access")
	flag.StringVar(&o.effort, "reasoning-effort", "", "reasoning effort: low, medium or high")
	flag.IntVar(&o.maxTurns, "max-turns", 10, "maximum model calls")
	flag.IntVar(&o.maxTokens, "max-tokens", 4096, "maximum output tokens per model call, including reasoning")
	flag.BoolVar(&o.stream, "stream", true, "stream model output")
	flag.Parse()
	if o.model == "" || flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: readfile -model MODEL [flags] PROMPT")
		os.Exit(2)
	}
	if err := run(o, strings.Join(flag.Args(), " ")); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func newClient(o options) (llm.Client, error) {
	keyEnv := o.keyEnv
	if keyEnv == "" {
		keyEnv = defaultKeyEnv[o.provider]
	}
	key := os.Getenv(keyEnv)
	switch o.provider {
	case "openai":
		opts := []openai.Option{openai.WithAPIKey(key)}
		if o.baseURL != "" {
			opts = append(opts, openai.WithBaseURL(o.baseURL))
		}
		return openai.New(opts...), nil
	case "chatcompletions":
		opts := []chatcompletions.Option{chatcompletions.WithAPIKey(key)}
		if o.baseURL != "" {
			opts = append(opts, chatcompletions.WithBaseURL(o.baseURL))
		}
		return chatcompletions.New(opts...), nil
	case "anthropic":
		opts := []anthropic.Option{anthropic.WithAPIKey(key)}
		if o.baseURL != "" {
			opts = append(opts, anthropic.WithBaseURL(o.baseURL))
		}
		return anthropic.New(opts...), nil
	}
	return nil, fmt.Errorf("unknown provider %q", o.provider)
}

func run(o options, prompt string) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	client, err := newClient(o)
	if err != nil {
		return err
	}
	cfg := agent.Config{
		Client:    client,
		Model:     o.model,
		System:    "You are a concise assistant.",
		MaxTurns:  o.maxTurns,
		MaxTokens: o.maxTokens,
		Logger:    slog.New(slog.NewTextHandler(os.Stderr, nil)),
	}
	if o.effort != "" {
		cfg.Reasoning = &llm.ReasoningConfig{Effort: o.effort}
	}
	files := readfile.New(o.root)
	defer files.Close()
	cfg.Plugins = []agent.Plugin{files}
	a, err := agent.New(ctx, cfg)
	if err != nil {
		return err
	}

	msgs := []llm.Message{llm.UserText(prompt)}
	var res *agent.Result
	if o.stream {
		for ev, err := range a.Stream(ctx, msgs) {
			if err != nil {
				return err
			}
			switch {
			case ev.Type == agent.EventModel && ev.LLM.Type == llm.EventTextDelta:
				fmt.Print(ev.LLM.Delta)
			case ev.Type == agent.EventModelResponse && ev.Response.Text() != "":
				fmt.Println()
			case ev.Type == agent.EventDone:
				res = ev.Result
			}
		}
	} else {
		if res, err = a.Run(ctx, msgs); err != nil {
			return err
		}
		fmt.Println(res.Text())
	}
	fmt.Fprintf(os.Stderr, "-- %s after %d turns, usage %+v\n", res.FinishReason, res.Turns, res.Usage)
	return nil
}

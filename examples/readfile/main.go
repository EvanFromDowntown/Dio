// Command readfile runs an agent that can read files under -root. Every run is
// recorded as a session that can be resumed.
//
//	OPENAI_API_KEY=... go run ./examples/readfile -model gpt-5 "Summarize README.md"
//	ANTHROPIC_API_KEY=... go run ./examples/readfile -provider anthropic/messages -model claude-sonnet-4-5 "Summarize README.md"
//	DIO_API_KEY=... go run ./examples/readfile -provider openai/chat-completions -api-key-env DIO_API_KEY \
//		-base-url https://gateway.example.com/v1 -model deepseek-flash "Summarize README.md"
//
//	go run ./examples/readfile -model gpt-5 -session 20261012-091500-a1b2c3 "And the next section?"
//	go run ./examples/readfile sessions
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/EvanFromDowntown/Dio/agent"
	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/llm/anthropic/messages"
	"github.com/EvanFromDowntown/Dio/llm/openai/chatcompletions"
	"github.com/EvanFromDowntown/Dio/llm/openai/responses"
	"github.com/EvanFromDowntown/Dio/plugins/readfile"
	"github.com/EvanFromDowntown/Dio/session"
)

var defaultKeyEnv = map[string]string{
	llm.ProviderResponses:       "OPENAI_API_KEY",
	llm.ProviderChatCompletions: "OPENAI_API_KEY",
	llm.ProviderMessages:        "ANTHROPIC_API_KEY",
}

type options struct {
	provider, model, baseURL, keyEnv, root, effort string
	sessionID, sessionsDir                         string
	newSession, stream                             bool
	maxTurns, maxTokens                            int
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "sessions" {
		if err := listSessions(os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		return
	}

	var o options
	flag.StringVar(&o.provider, "provider", llm.ProviderResponses, "provider ID: "+strings.Join(llm.Providers, ", "))
	flag.StringVar(&o.model, "model", "", "model name (required)")
	flag.StringVar(&o.baseURL, "base-url", "", "override the provider base URL")
	flag.StringVar(&o.keyEnv, "api-key-env", "", "environment variable holding the API key (default depends on -provider)")
	flag.StringVar(&o.root, "root", ".", "directory read_file may access")
	flag.StringVar(&o.effort, "reasoning-effort", "", "reasoning effort: low, medium or high")
	flag.IntVar(&o.maxTurns, "max-turns", 10, "maximum model calls")
	flag.IntVar(&o.maxTokens, "max-tokens", 4096, "maximum output tokens per model call, including reasoning")
	flag.BoolVar(&o.stream, "stream", true, "stream model output")
	flag.StringVar(&o.sessionID, "session", "", "resume this session instead of starting a new one")
	flag.BoolVar(&o.newSession, "new", false, "start a new session (the default)")
	flag.StringVar(&o.sessionsDir, "sessions-dir", "", "session directory (default ~/.dio/sessions)")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: readfile -model MODEL [flags] PROMPT\n       readfile sessions [-sessions-dir DIR]")
		flag.PrintDefaults()
	}
	flag.Parse()
	if o.model == "" || (flag.NArg() == 0 && o.sessionID == "") {
		flag.Usage()
		os.Exit(2)
	}
	if o.newSession && o.sessionID != "" {
		fmt.Fprintln(os.Stderr, "error: -new and -session are mutually exclusive")
		os.Exit(2)
	}
	if err := run(o, strings.Join(flag.Args(), " ")); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func openStore(dir string) (*session.FileStore, error) {
	if dir == "" {
		var err error
		if dir, err = session.DefaultDir(); err != nil {
			return nil, err
		}
	}
	return session.NewFileStore(dir)
}

func listSessions(args []string) error {
	fs := flag.NewFlagSet("sessions", flag.ExitOnError)
	dir := fs.String("sessions-dir", "", "session directory (default ~/.dio/sessions)")
	fs.Parse(args)
	store, err := openStore(*dir)
	if err != nil {
		return err
	}
	infos, err := store.List(context.Background())
	if err != nil {
		return err
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "ID\tUPDATED\tMESSAGES\tTOKENS(in/out)\tPROVIDER/MODEL\tTITLE")
	for _, s := range infos {
		fmt.Fprintf(w, "%s\t%s\t%d\t%d/%d\t%s/%s\t%s\n", s.ID, s.UpdatedAt.Local().Format(time.DateTime),
			s.MessageCount, s.Usage.InputTokens, s.Usage.OutputTokens, s.Provider, s.Model, s.Title)
	}
	return w.Flush()
}

func newClient(o options) (llm.Client, error) {
	provider := llm.NormalizeProvider(o.provider)
	keyEnv := o.keyEnv
	if keyEnv == "" {
		keyEnv = defaultKeyEnv[provider]
	}
	key := os.Getenv(keyEnv)
	switch provider {
	case llm.ProviderResponses:
		opts := []responses.Option{responses.WithAPIKey(key)}
		if o.baseURL != "" {
			opts = append(opts, responses.WithBaseURL(o.baseURL))
		}
		return responses.New(opts...), nil
	case llm.ProviderChatCompletions:
		opts := []chatcompletions.Option{chatcompletions.WithAPIKey(key)}
		if o.baseURL != "" {
			opts = append(opts, chatcompletions.WithBaseURL(o.baseURL))
		}
		return chatcompletions.New(opts...), nil
	case llm.ProviderMessages:
		opts := []messages.Option{messages.WithAPIKey(key)}
		if o.baseURL != "" {
			opts = append(opts, messages.WithBaseURL(o.baseURL))
		}
		return messages.New(opts...), nil
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
	store, err := openStore(o.sessionsDir)
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
	cfg.Plugins = []agent.Plugin{files, session.NewRecorder(store)}
	a, err := agent.New(ctx, cfg)
	if err != nil {
		return err
	}

	id := o.sessionID
	if id == "" {
		sess, err := store.Create(ctx, session.Info{Provider: llm.NormalizeProvider(o.provider), Model: o.model})
		if err != nil {
			return err
		}
		id = sess.ID
	}
	fmt.Fprintf(os.Stderr, "-- session %s\n", id)

	var input []llm.Message
	if prompt != "" {
		input = append(input, llm.UserText(prompt))
	}
	runner := session.Runner{Agent: a, Store: store}
	var res *agent.Result
	if o.stream {
		for ev, evErr := range runner.Stream(ctx, id, input...) {
			switch {
			case ev.Type == agent.EventModel && ev.LLM.Type == llm.EventTextDelta:
				fmt.Print(ev.LLM.Delta)
			case ev.Type == agent.EventModelResponse && ev.Response.Text() != "":
				fmt.Println()
			case ev.Type == agent.EventDone:
				res, err = ev.Result, evErr
			}
		}
	} else {
		res, err = runner.Run(ctx, id, input...)
		if res != nil {
			fmt.Println(res.Text())
		}
	}
	if res != nil {
		fmt.Fprintf(os.Stderr, "-- %s after %d turns, usage %+v\n", res.FinishReason, res.Turns, res.Usage)
	}
	if err != nil {
		if !errors.Is(err, session.ErrNotFound) {
			fmt.Fprintf(os.Stderr, "-- history saved; resume with -session %s\n", id)
		}
		return err
	}
	return nil
}

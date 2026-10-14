package cli

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/EvanFromDowntown/Dio/agent"
	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/plugins/readfile"
	"github.com/EvanFromDowntown/Dio/session"
)

const SystemPrompt = "You are Dio, a concise and helpful assistant. Answer in the user's language."

// App is one dio process: an agent bound to a session store, and the
// session currently being chatted in.
type App struct {
	opts   *Options
	agent  *agent.Agent
	store  session.Store
	runner session.Runner
	files  *readfile.Plugin

	// Out receives answers; Info receives tool activity and status lines.
	Out, Info io.Writer

	sessionID string
	resumed   *session.Session
}

// NewApp builds the agent. When opts.SessionID is set the session must
// exist; it is only created lazily for new conversations, on the first
// message.
func NewApp(ctx context.Context, opts *Options, client llm.Client, store session.Store, out, info io.Writer) (*App, error) {
	a := &App{opts: opts, store: store, Out: out, Info: info}
	if opts.SessionID != "" {
		sess, err := store.Get(ctx, opts.SessionID)
		if err != nil {
			if errors.Is(err, session.ErrNotFound) {
				return nil, fmt.Errorf("会话 %s 不存在（用 dio sessions 查看）", opts.SessionID)
			}
			return nil, err
		}
		a.sessionID, a.resumed = sess.ID, sess
	}
	if opts.Model == "" {
		return nil, errors.New("请用 -model 或环境变量 DIO_MODEL 指定模型")
	}

	cfg := agent.Config{
		Client:      client,
		Model:       opts.Model,
		System:      SystemPrompt,
		MaxTurns:    opts.MaxTurns,
		MaxTokens:   opts.MaxTokens,
		Temperature: opts.Temperature,
	}
	if opts.ReasoningEffort != "" {
		cfg.Reasoning = &llm.ReasoningConfig{Effort: opts.ReasoningEffort}
	}
	a.files = readfile.New(opts.Root)
	cfg.Plugins = []agent.Plugin{a.files, session.NewRecorder(store)}
	ag, err := agent.New(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("初始化失败：%w", err)
	}
	a.agent = ag
	a.runner = session.Runner{Agent: ag, Store: store}
	return a, nil
}

// ApplySessionDefaults fills provider, model and base URL from the session
// being resumed when they were not given explicitly.
func ApplySessionDefaults(ctx context.Context, opts *Options, store session.Store) error {
	if opts.SessionID == "" {
		return nil
	}
	sess, err := store.Get(ctx, opts.SessionID)
	if err != nil {
		if errors.Is(err, session.ErrNotFound) {
			return fmt.Errorf("会话 %s 不存在（用 dio sessions 查看）", opts.SessionID)
		}
		return err
	}
	if !opts.Explicit("provider") && contains(Providers, sess.Provider) {
		opts.Provider = sess.Provider
	}
	if !opts.Explicit("model") && sess.Model != "" {
		opts.Model = sess.Model
	}
	if !opts.Explicit("base-url") && sess.Metadata["base_url"] != "" {
		opts.BaseURL = sess.Metadata["base_url"]
	}
	return nil
}

func (a *App) Close() error { return a.files.Close() }

func (a *App) SessionID() string { return a.sessionID }

func (a *App) ensureSession(ctx context.Context) error {
	if a.sessionID != "" {
		return nil
	}
	meta := map[string]string{}
	if root, err := filepath.Abs(a.opts.Root); err == nil {
		meta["root"] = root
	}
	if a.opts.BaseURL != "" {
		meta["base_url"] = a.opts.BaseURL
	}
	sess, err := a.store.Create(ctx, session.Info{Provider: a.opts.Provider, Model: a.opts.Model, Metadata: meta})
	if err != nil {
		return fmt.Errorf("创建会话失败：%w", err)
	}
	a.sessionID = sess.ID
	return nil
}

// Ask sends one user message in the current session and streams the answer.
func (a *App) Ask(ctx context.Context, text string) (*agent.Result, error) {
	if err := a.ensureSession(ctx); err != nil {
		return nil, err
	}
	r := NewRenderer(a.Out, a.Info, a.opts.ShowReasoning)
	var (
		res *agent.Result
		err error
	)
	for ev, evErr := range a.runner.Stream(ctx, a.sessionID, llm.UserText(text)) {
		r.Event(ev)
		if ev.Type == agent.EventDone {
			res, err = ev.Result, evErr
		}
	}
	r.Finish()
	if res != nil && res.Response != nil {
		fmt.Fprintln(a.Info, FormatUsage(res))
	}
	if err != nil {
		switch {
		case res != nil && res.FinishReason == agent.FinishCanceled:
			fmt.Fprintf(a.Info, "已中断。历史已保存，可用 dio -session %s 继续。\n", a.sessionID)
		case res != nil && res.FinishReason == agent.FinishMaxTurns:
			fmt.Fprintf(a.Info, "已达到 -max-turns 上限（%d）。历史已保存，发送新消息或用 dio -session %s 继续。\n", a.opts.MaxTurns, a.sessionID)
		default:
			fmt.Fprintf(a.Info, "出错：%v\n历史已保存，可用 dio -session %s 继续。\n", err, a.sessionID)
		}
	}
	return res, err
}

func (a *App) ListSessions(ctx context.Context, w io.Writer) error {
	infos, err := a.store.List(ctx)
	if err != nil {
		return err
	}
	return WriteSessions(w, infos, a.sessionID)
}

// OneShot answers a single message: prompt if non-empty, otherwise all of in.
func (a *App) OneShot(ctx context.Context, in io.Reader, prompt string) error {
	if prompt == "" {
		data, err := io.ReadAll(in)
		if err != nil {
			return fmt.Errorf("读取标准输入失败：%w", err)
		}
		prompt = strings.TrimSpace(string(data))
	}
	if prompt == "" {
		return errors.New("没有输入：请通过参数或标准输入提供问题")
	}
	_, err := a.Ask(ctx, prompt)
	if err == nil {
		fmt.Fprintf(a.Info, "会话 %s 已保存，可用 dio -session %s 继续。\n", a.sessionID, a.sessionID)
	}
	return err
}

// REPL reads lines from in until /quit, EOF or an interrupt at the prompt.
// An interrupt while a message is being answered cancels only that turn.
func (a *App) REPL(ctx context.Context, in io.Reader, interrupts <-chan os.Signal) error {
	fmt.Fprintf(a.Out, "Dio · %s/%s · 工作目录 %s\n", a.opts.Provider, a.opts.Model, a.opts.Root)
	if a.resumed != nil {
		fmt.Fprintf(a.Out, "已恢复会话 %s（%d 条消息）：%s\n", a.resumed.ID, a.resumed.MessageCount, a.resumed.Title)
	}
	fmt.Fprintln(a.Out, "输入消息开始对话，/help 查看命令，Ctrl+C 中断生成。")

	lines := make(chan string)
	go func() {
		defer close(lines)
		sc := bufio.NewScanner(in)
		sc.Buffer(make([]byte, 64<<10), 4<<20)
		for sc.Scan() {
			select {
			case lines <- sc.Text():
			case <-ctx.Done():
				return
			}
		}
	}()

	for {
		fmt.Fprint(a.Out, "\n你> ")
		var line string
		select {
		case l, ok := <-lines:
			if !ok {
				fmt.Fprintln(a.Out)
				a.goodbye()
				return nil
			}
			line = l
		case <-interrupts:
			fmt.Fprintln(a.Out)
			a.goodbye()
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}

		cmd := ParseCommand(line)
		switch cmd.Kind {
		case CmdEmpty:
		case CmdQuit:
			a.goodbye()
			return nil
		case CmdHelp:
			fmt.Fprintln(a.Out, helpText)
		case CmdSessions:
			if err := a.ListSessions(ctx, a.Out); err != nil {
				fmt.Fprintf(a.Info, "列出会话失败：%v\n", err)
			}
		case CmdNew:
			if a.sessionID != "" {
				fmt.Fprintf(a.Out, "已离开会话 %s，下一条消息将开始新会话。\n", a.sessionID)
			} else {
				fmt.Fprintln(a.Out, "下一条消息将开始新会话。")
			}
			a.sessionID, a.resumed = "", nil
		case CmdUnknown:
			fmt.Fprintf(a.Out, "未知命令 %s，输入 /help 查看可用命令。\n", cmd.Text)
		case CmdMessage:
			a.askInterruptible(ctx, cmd.Text, interrupts)
		}
	}
}

func (a *App) askInterruptible(ctx context.Context, text string, interrupts <-chan os.Signal) {
	turnCtx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	go func() {
		select {
		case <-interrupts:
			cancel()
		case <-done:
		}
	}()
	a.Ask(turnCtx, text)
	close(done)
	cancel()
}

func (a *App) goodbye() {
	if a.sessionID != "" {
		fmt.Fprintf(a.Out, "再见。会话已保存，继续请用：dio -session %s\n", a.sessionID)
	} else {
		fmt.Fprintln(a.Out, "再见。")
	}
}

package cli

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/EvanFromDowntown/Dio/agent"
	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/session"
)

func assertContains(t *testing.T, label, got string, wants ...string) {
	t.Helper()
	for _, w := range wants {
		if !strings.Contains(got, w) {
			t.Errorf("%s missing %q:\n%s", label, w, got)
		}
	}
}

func TestOneShotWithToolCall(t *testing.T) {
	h := newHarness(t, nil, nil, readFile("notes.txt"), answer("代号是 Falcon-7。"))
	input := "  记住：我的柜子密码是 4729。然后读 notes.txt 告诉我代号。\n"
	if err := h.app.OneShot(context.Background(), strings.NewReader(input), ""); err != nil {
		t.Fatal(err)
	}
	if got := h.out.String(); got != "让我看看。\n代号是 Falcon-7。\n" {
		t.Errorf("stdout = %q", got)
	}
	assertContains(t, "info", h.info.String(),
		"[工具] read_file(path=notes.txt)",
		"[结果] read_file：46 字节 · Project codename: Falcon-7",
		"（2 次模型调用 · 输入 180 / 输出 15 tokens · 缓存命中 50）",
		"会话 "+h.app.SessionID()+" 已保存",
	)
	if strings.Contains(h.info.String()+h.out.String(), "secret thoughts") {
		t.Error("reasoning shown without -show-reasoning")
	}

	sess, err := h.store.Get(context.Background(), h.app.SessionID())
	if err != nil {
		t.Fatal(err)
	}
	if len(sess.Messages) != 4 || sess.Messages[0].Text() != strings.TrimSpace(input) ||
		sess.Provider != DefaultProvider || sess.Model != "fake-model" || sess.Metadata["root"] != h.opts.Root {
		t.Errorf("session = %+v", sess)
	}
	req := h.client.requests[0]
	if req.MaxTokens != 4096 || len(req.Tools) != 1 || req.Tools[0].Name != "read_file" || !strings.Contains(req.System, "read_file") {
		t.Errorf("request = %+v", req)
	}
}

func TestOneShotErrors(t *testing.T) {
	h := newHarness(t, nil, nil)
	if err := h.app.OneShot(context.Background(), strings.NewReader(" \n"), ""); err == nil || !strings.Contains(err.Error(), "没有输入") {
		t.Errorf("empty input: %v", err)
	}
	if list, _ := h.store.List(context.Background()); len(list) != 0 {
		t.Errorf("empty input created a session: %+v", list)
	}

	h = newHarness(t, nil, nil, func(context.Context, *llm.Request) (*llm.Response, error) {
		return nil, &llm.APIError{Provider: "fake", StatusCode: 503, Message: "upstream down"}
	})
	err := h.app.OneShot(context.Background(), nil, "你好")
	if !llm.IsRetryable(err) {
		t.Errorf("err = %v", err)
	}
	assertContains(t, "info", h.info.String(), "出错：", "dio -session "+h.app.SessionID())
	if strings.Contains(h.info.String(), "次模型调用") {
		t.Errorf("usage footer printed although no model call succeeded:\n%s", h.info.String())
	}
}

func TestREPLMultiTurnAndCommands(t *testing.T) {
	store := session.NewMemoryStore()
	h := newHarness(t, store, []string{"-temperature", "0.5", "-reasoning-effort", "high", "-max-tokens", "1000"},
		answer("好的，记住了。"), readFile("notes.txt"), answer("4729，代号 Falcon-7。"), answer("新会话你好。")).interactive()
	input := "记住 4729\n\n   \n/sessions\n/bogus\n密码是多少？顺便读 notes.txt\n/new\n你好\n/quit\n不会被发送\n"
	if err := h.app.REPL(context.Background(), strings.NewReader(input), nil); err != nil {
		t.Fatal(err)
	}
	out := h.out.String()
	assertContains(t, "output", out,
		"Dio · chatcompletions/fake-model",
		"你> 好的，记住了。\n",
		"未知命令 /bogus",
		"[工具] read_file(path=notes.txt)\n[结果] read_file：46 字节",
		"4729，代号 Falcon-7。\n（",
		"下一条消息将开始新会话",
		"再见。会话已保存，继续请用：dio -session "+h.app.SessionID(),
	)
	if n := strings.Count(out, "\n你> "); n != 9 {
		t.Errorf("expected 9 prompts, got %d:\n%s", n, out)
	}

	infos, _ := store.List(context.Background())
	if len(infos) != 2 {
		t.Fatalf("sessions = %+v", infos)
	}
	first, err := store.Get(context.Background(), infos[1].ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Messages) != 6 || first.Title != "记住 4729" {
		t.Errorf("first session: %d messages, title %q", len(first.Messages), first.Title)
	}
	assertContains(t, "/sessions output", out, "* "+first.ID)

	req := h.client.requests[1]
	if len(req.Messages) != 3 || req.Messages[0].Text() != "记住 4729" || *req.Temperature != 0.5 ||
		req.Reasoning.Effort != "high" || req.MaxTokens != 1000 {
		t.Errorf("turn 2 request = %+v", req)
	}
	if last := h.client.lastRequest(); len(last.Messages) != 1 || last.Messages[0].Text() != "你好" {
		t.Errorf("/new did not start a fresh history: %+v", last.Messages)
	}
}

func TestResumeSession(t *testing.T) {
	store := session.NewMemoryStore()
	h := newHarness(t, store, []string{"-provider", "openai", "-model", "orig-model", "-base-url", "https://gw/v1"}, answer("记住了。"))
	if err := h.app.OneShot(context.Background(), nil, "记住 4729"); err != nil {
		t.Fatal(err)
	}
	id := h.app.SessionID()

	resumed := newHarness(t, store, []string{"-session", id}, answer("是 4729。")).interactive()
	if resumed.opts.Provider != "openai" || resumed.opts.Model != "orig-model" || resumed.opts.BaseURL != "https://gw/v1" {
		t.Errorf("session defaults not applied: %+v", resumed.opts)
	}
	if err := resumed.app.REPL(context.Background(), strings.NewReader("密码？\n"), nil); err != nil {
		t.Fatal(err)
	}
	assertContains(t, "banner", resumed.out.String(), "已恢复会话 "+id+"（2 条消息）：记住 4729")
	req := resumed.client.lastRequest()
	if len(req.Messages) != 3 || req.Messages[0].Text() != "记住 4729" || req.Messages[2].Text() != "密码？" {
		t.Errorf("resumed request = %+v", req.Messages)
	}
	if sess, _ := store.Get(context.Background(), id); len(sess.Messages) != 4 {
		t.Errorf("resumed session has %d messages", len(sess.Messages))
	}

	explicit := newHarness(t, store, []string{"-session", id, "-provider", "anthropic"})
	if explicit.opts.Provider != "anthropic" || explicit.opts.Model != "orig-model" {
		t.Errorf("explicit flags must win: %+v", explicit.opts)
	}

	opts, _ := Parse([]string{"-session", "nope"}, noEnv, io.Discard)
	if err := ApplySessionDefaults(context.Background(), opts, store); err == nil || !strings.Contains(err.Error(), "不存在") {
		t.Errorf("missing session: %v", err)
	}
}

func TestREPLInterruptCancelsTurnOnly(t *testing.T) {
	interrupts := make(chan os.Signal, 1)
	h := newHarness(t, nil, nil,
		func(ctx context.Context, _ *llm.Request) (*llm.Response, error) {
			interrupts <- os.Interrupt
			<-ctx.Done()
			return nil, ctx.Err()
		},
		answer("继续了。"),
	).interactive()
	input := strings.NewReader("写一篇长文\n还在吗？\n")
	if err := h.app.REPL(context.Background(), input, interrupts); err != nil {
		t.Fatal(err)
	}
	id := h.app.SessionID()
	assertContains(t, "output", h.out.String(), "已中断。历史已保存，可用 dio -session "+id+" 继续。", "继续了。")

	sess, err := h.store.Get(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, m := range sess.Messages {
		got = append(got, m.Text())
	}
	if want := []string{"写一篇长文", "还在吗？", "继续了。"}; !reflect.DeepEqual(got, want) {
		t.Errorf("history = %q", got)
	}
}

func TestREPLInterruptAtPromptExits(t *testing.T) {
	h := newHarness(t, nil, nil).interactive()
	interrupts := make(chan os.Signal, 1)
	interrupts <- os.Interrupt
	blocked, w := io.Pipe()
	defer w.Close()
	if err := h.app.REPL(context.Background(), blocked, interrupts); err != nil {
		t.Fatal(err)
	}
	assertContains(t, "output", h.out.String(), "再见。")
}

func TestMaxTurnsThenResume(t *testing.T) {
	h := newHarness(t, nil, []string{"-max-turns", "1"}, readFile("notes.txt"), answer("读完了。"))
	err := h.app.OneShot(context.Background(), nil, "读 notes.txt")
	if !errors.Is(err, agent.ErrMaxTurns) {
		t.Fatalf("err = %v", err)
	}
	assertContains(t, "info", h.info.String(), "已达到 -max-turns 上限（1）")

	opts, _ := Parse([]string{"-model", "fake-model", "-root", h.opts.Root, "-session", h.app.SessionID()}, noEnv, io.Discard)
	app, err := NewApp(context.Background(), opts, h.client, h.store, io.Discard, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	defer app.Close()
	if _, err := app.runner.Run(context.Background(), app.SessionID()); err != nil {
		t.Fatal(err)
	}
	sess, _ := h.store.Get(context.Background(), app.SessionID())
	if sess.Messages[len(sess.Messages)-1].Text() != "读完了。" {
		t.Errorf("resume did not finish: %+v", sess.Messages)
	}
}

func TestNewClient(t *testing.T) {
	opts, _ := Parse([]string{"-api-key-env", "MY_KEY"}, noEnv, io.Discard)
	if _, err := NewClient(opts, noEnv); err == nil || !strings.Contains(err.Error(), "MY_KEY") {
		t.Errorf("missing key: %v", err)
	}
	withKey := func(k string) string { return map[string]string{"DIO_API_KEY": "k"}[k] }
	for _, p := range Providers {
		opts, _ := Parse([]string{"-provider", p, "-base-url", "https://gw/v1/"}, noEnv, io.Discard)
		if c, err := NewClient(opts, withKey); err != nil || c == nil {
			t.Errorf("%s: %v", p, err)
		}
	}
}

func TestNewAppErrors(t *testing.T) {
	store := session.NewMemoryStore()
	opts, _ := Parse(nil, noEnv, io.Discard)
	if _, err := NewApp(context.Background(), opts, &fakeClient{}, store, io.Discard, io.Discard); err == nil || !strings.Contains(err.Error(), "-model") {
		t.Errorf("missing model: %v", err)
	}
	opts, _ = Parse([]string{"-model", "m", "-root", filepath.Join(t.TempDir(), "missing")}, noEnv, io.Discard)
	if _, err := NewApp(context.Background(), opts, &fakeClient{}, store, io.Discard, io.Discard); err == nil {
		t.Error("missing root should fail")
	}
	opts, _ = Parse([]string{"-model", "m", "-session", "nope"}, noEnv, io.Discard)
	if _, err := NewApp(context.Background(), opts, &fakeClient{}, store, io.Discard, io.Discard); err == nil {
		t.Error("missing session should fail")
	}
}

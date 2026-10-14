package webui

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/EvanFromDowntown/Dio/llm"
)

func TestIndexPage(t *testing.T) {
	e := newEnv(t, testKey)
	resp, err := e.http.Client().Get(e.http.URL + "/")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 || !strings.Contains(string(body), "<title>Dio</title>") ||
		!strings.Contains(string(body), "新建会话") || resp.Header.Get("Content-Security-Policy") == "" {
		t.Errorf("index: %d %s", resp.StatusCode, resp.Header)
	}
	if code := e.do("GET", "/nope", nil, nil); code != http.StatusNotFound {
		t.Errorf("unknown path: %d", code)
	}
}

func TestConfig(t *testing.T) {
	e := newEnv(t, testKey)
	raw := func() string {
		resp, err := e.http.Client().Get(e.http.URL + "/api/config")
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		return string(b)
	}
	var cfg configView
	e.do("GET", "/api/config", nil, &cfg)
	if !cfg.HasAPIKey || cfg.Provider != "chatcompletions" || cfg.Model != "fake-model" || cfg.Root != e.root ||
		cfg.Temperature != nil || len(cfg.Providers) != 3 {
		t.Errorf("config = %+v", cfg)
	}
	if strings.Contains(raw(), testKey) {
		t.Fatal("config response leaks the API key")
	}

	update := map[string]any{
		"provider": "anthropic", "model": "m2", "base_url": "https://other/v1", "root": e.root,
		"max_turns": 3, "max_tokens": 222, "reasoning_effort": "high", "temperature": 0.7, "api_key": "sk-new-key",
	}
	if code := e.do("PUT", "/api/config", update, &cfg); code != 200 {
		t.Fatalf("put: %d", code)
	}
	if cfg.Provider != "anthropic" || cfg.MaxTokens != 222 || *cfg.Temperature != 0.7 || cfg.ReasoningEffort != "high" || !cfg.HasAPIKey {
		t.Errorf("updated = %+v", cfg)
	}
	if strings.Contains(raw(), "sk-new-key") {
		t.Fatal("config response leaks the new API key")
	}

	update["api_key"] = ""
	update["temperature"] = nil
	e.do("PUT", "/api/config", update, &cfg)
	if !cfg.HasAPIKey || cfg.Temperature != nil {
		t.Errorf("empty api_key must keep the key, null temperature must reset: %+v", cfg)
	}
	e.srv.mu.Lock()
	key := e.srv.apiKey
	e.srv.mu.Unlock()
	if key != "sk-new-key" {
		t.Errorf("key = %q", key)
	}

	update["clear_api_key"] = true
	e.do("PUT", "/api/config", update, &cfg)
	if cfg.HasAPIKey {
		t.Error("clear_api_key did not clear")
	}

	for name, bad := range map[string]map[string]any{
		"provider":    {"provider": "gemini", "max_turns": 1, "max_tokens": 1},
		"effort":      {"reasoning_effort": "max", "max_turns": 1, "max_tokens": 1},
		"temperature": {"temperature": 5, "max_turns": 1, "max_tokens": 1},
		"turns":       {"max_turns": 0, "max_tokens": 1},
		"root":        {"root": e.root + "/missing", "max_turns": 1, "max_tokens": 1},
	} {
		var out map[string]string
		if code := e.do("PUT", "/api/config", bad, &out); code != http.StatusBadRequest || out["error"] == "" {
			t.Errorf("%s: %d %v", name, code, out)
		}
	}
}

func TestChatFlowWithToolsAndResume(t *testing.T) {
	e := newEnv(t, testKey, readFile("notes.txt"), answer("代号是 Falcon-7。"), answer("你的停车位是 B2-117。"))
	sess := e.createSession()
	if sess.Provider != "chatcompletions" || sess.Model != "fake-model" {
		t.Errorf("created = %+v", sess)
	}

	events := e.chat(sess.ID, "记住：停车位 B2-117。然后读 notes.txt")
	var names []string
	var text strings.Builder
	var items []Item
	for _, ev := range events {
		names = append(names, ev.Name)
		switch ev.Name {
		case "text":
			var d deltaEvent
			json.Unmarshal(ev.Data, &d)
			text.WriteString(d.Delta)
		case "item":
			var it Item
			json.Unmarshal(ev.Data, &it)
			items = append(items, it)
		}
	}
	wantNames := []string{"text", "text", "message_end", "item", "item", "reasoning", "text", "text", "message_end", "done"}
	if !reflect.DeepEqual(names, wantNames) {
		t.Errorf("events = %v", names)
	}
	if text.String() != "让我看看。代号是 Falcon-7。" {
		t.Errorf("text = %q", text.String())
	}
	wantItems := []Item{
		{Kind: "tool_call", ToolID: "c-notes.txt", Name: "read_file", Args: "path=notes.txt"},
		{Kind: "tool_result", ToolID: "c-notes.txt", Name: "read_file", Bytes: 46, Preview: "Project codename: Falcon-7"},
	}
	if !reflect.DeepEqual(items, wantItems) {
		t.Errorf("items = %+v", items)
	}
	done := doneOf(t, events)
	if done.FinishReason != "completed" || done.Turns != 2 || done.Usage.InputTokens != 180 || done.Error != "" ||
		done.Title != "记住：停车位 B2-117。然后读 notes.txt" || done.SessionID != sess.ID {
		t.Errorf("done = %+v", done)
	}

	req := e.client.calls()[0]
	if req.Model != "fake-model" || req.MaxTokens != 1000 || len(req.Tools) != 1 || req.Tools[0].Name != "read_file" {
		t.Errorf("request = %+v", req)
	}
	e.factory.mu.Lock()
	if e.factory.calls[0] != "chatcompletions|https://gw/v1|"+testKey {
		t.Errorf("factory = %v", e.factory.calls)
	}
	e.factory.mu.Unlock()

	// Second message in the same session carries the history.
	if d := doneOf(t, e.chat(sess.ID, "我的停车位是多少？")); d.Error != "" {
		t.Fatalf("second message: %+v", d)
	}
	last := e.client.calls()[2]
	if len(last.Messages) != 5 || last.Messages[0].Text() != "记住：停车位 B2-117。然后读 notes.txt" || last.Messages[4].Text() != "我的停车位是多少？" {
		t.Errorf("second request history = %+v", last.Messages)
	}

	var list []SessionSummary
	e.do("GET", "/api/sessions", nil, &list)
	if len(list) != 1 || list[0].MessageCount != 6 || list[0].Title == "" || list[0].Usage.OutputTokens != 25 {
		t.Errorf("list = %+v", list)
	}

	var view sessionView
	if code := e.do("GET", "/api/sessions/"+sess.ID, nil, &view); code != 200 {
		t.Fatalf("get session: %d", code)
	}
	var kinds []string
	for _, it := range view.Items {
		kinds = append(kinds, it.Kind)
	}
	wantKinds := []string{"user", "assistant", "tool_call", "tool_result", "reasoning", "assistant", "user", "reasoning", "assistant"}
	if !reflect.DeepEqual(kinds, wantKinds) || view.Items[3].Name != "read_file" || view.Items[8].Text != "你的停车位是 B2-117。" || view.Running {
		t.Errorf("history kinds = %v\nitems = %+v", kinds, view.Items)
	}
}

func TestMissingKeyAndModel(t *testing.T) {
	e := newEnv(t, "")
	sess := e.createSession()
	var out map[string]string
	b, _ := json.Marshal(map[string]string{"text": "你好"})
	resp, _ := e.http.Client().Post(e.http.URL+"/api/sessions/"+sess.ID+"/messages", "application/json", bytes.NewReader(b))
	json.NewDecoder(resp.Body).Decode(&out)
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest || !strings.Contains(out["error"], "API key") {
		t.Errorf("missing key: %d %v", resp.StatusCode, out)
	}
	if len(e.client.calls()) != 0 {
		t.Error("model called without a key")
	}

	e.do("PUT", "/api/config", map[string]any{"model": "", "root": e.root, "max_turns": 1, "max_tokens": 1, "api_key": "k"}, nil)
	resp, _ = e.http.Client().Post(e.http.URL+"/api/sessions/"+sess.ID+"/messages", "application/json", bytes.NewReader(b))
	json.NewDecoder(resp.Body).Decode(&out)
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest || !strings.Contains(out["error"], "模型") {
		t.Errorf("missing model: %d %v", resp.StatusCode, out)
	}
}

func TestRequestValidation(t *testing.T) {
	e := newEnv(t, testKey)
	sess := e.createSession()
	var out map[string]string
	if code := e.do("POST", "/api/sessions/nope/messages", map[string]string{"text": "x"}, &out); code != http.StatusNotFound {
		t.Errorf("unknown session post: %d", code)
	}
	if code := e.do("GET", "/api/sessions/nope", nil, &out); code != http.StatusNotFound || out["error"] != "会话不存在" {
		t.Errorf("unknown session get: %d %v", code, out)
	}
	if code := e.do("GET", "/api/sessions/..%2Fetc", nil, nil); code != http.StatusNotFound {
		t.Errorf("traversal id: %d", code)
	}
	if code := e.do("POST", "/api/sessions/"+sess.ID+"/messages", map[string]string{"text": "  "}, &out); code != http.StatusBadRequest {
		t.Errorf("empty text: %d", code)
	}

	post := func(mutate func(*http.Request)) int {
		req, _ := http.NewRequest("POST", e.http.URL+"/api/sessions", strings.NewReader("{}"))
		req.Header.Set("Content-Type", "application/json")
		mutate(req)
		resp, err := e.http.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}
	if code := post(func(r *http.Request) { r.Header.Set("Content-Type", "text/plain") }); code != http.StatusUnsupportedMediaType {
		t.Errorf("form post: %d", code)
	}
	if code := post(func(r *http.Request) { r.Header.Set("Origin", "https://evil.example") }); code != http.StatusForbidden {
		t.Errorf("cross-origin: %d", code)
	}
	if code := post(func(r *http.Request) { r.Host = "evil.example:8080" }); code != http.StatusForbidden {
		t.Errorf("rebinding host: %d", code)
	}
	if code := post(func(r *http.Request) { r.Header.Set("Origin", e.http.URL) }); code != http.StatusCreated {
		t.Errorf("same-origin: %d", code)
	}
	for host, want := range map[string]bool{"127.0.0.1:8080": true, "localhost:1": true, "[::1]:80": true, "LOCALHOST": true, "192.168.1.2:8080": false, "dio.local": false} {
		if isLoopbackHost(host) != want {
			t.Errorf("isLoopbackHost(%q) != %v", host, want)
		}
	}
}

func TestStopAndConflict(t *testing.T) {
	started := make(chan struct{})
	e := newEnv(t, testKey,
		func(ctx context.Context, _ *llm.Request) (*llm.Response, error) {
			close(started)
			<-ctx.Done()
			return nil, ctx.Err()
		},
		answer("继续了。"),
	)
	sess := e.createSession()
	result := make(chan []sseEvent)
	go func() { result <- e.chat(sess.ID, "写一篇长文") }()
	<-started

	var view sessionView
	e.do("GET", "/api/sessions/"+sess.ID, nil, &view)
	if !view.Running {
		t.Error("session not marked running")
	}
	var out map[string]any
	if code := e.do("POST", "/api/sessions/"+sess.ID+"/messages", map[string]string{"text": "再来"}, &out); code != http.StatusConflict {
		t.Errorf("concurrent run: %d %v", code, out)
	}
	if code := e.do("POST", "/api/sessions/"+sess.ID+"/cancel", map[string]any{}, &out); code != 200 || out["canceled"] != true {
		t.Errorf("cancel: %d %v", code, out)
	}
	done := doneOf(t, <-result)
	if done.FinishReason != "canceled" || !strings.Contains(done.Error, "已停止") {
		t.Errorf("done = %+v", done)
	}

	// The interrupted message is saved and the session continues.
	if d := doneOf(t, e.chat(sess.ID, "还在吗？")); d.Error != "" {
		t.Fatalf("after stop: %+v", d)
	}
	e.do("GET", "/api/sessions/"+sess.ID, nil, &view)
	var texts []string
	for _, it := range view.Items {
		if it.Kind == "user" || it.Kind == "assistant" {
			texts = append(texts, it.Text)
		}
	}
	if want := []string{"写一篇长文", "还在吗？", "继续了。"}; !reflect.DeepEqual(texts, want) {
		t.Errorf("history = %q", texts)
	}
}

func TestBrowserDisconnectCancelsRun(t *testing.T) {
	started, canceled := make(chan struct{}), make(chan struct{})
	e := newEnv(t, testKey, func(ctx context.Context, _ *llm.Request) (*llm.Response, error) {
		close(started)
		<-ctx.Done()
		close(canceled)
		return nil, ctx.Err()
	})
	sess := e.createSession()
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, "POST", e.http.URL+"/api/sessions/"+sess.ID+"/messages", strings.NewReader(`{"text":"hi"}`))
	req.Header.Set("Content-Type", "application/json")
	resp, err := e.http.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	<-started
	cancel()
	resp.Body.Close()
	select {
	case <-canceled:
	case <-time.After(5 * time.Second):
		t.Fatal("run not canceled after the browser disconnected")
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		var view sessionView
		e.do("GET", "/api/sessions/"+sess.ID, nil, &view)
		if !view.Running {
			if len(view.Items) != 1 || view.Items[0].Text != "hi" {
				t.Errorf("history = %+v", view.Items)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("run still marked running")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestModelErrorIsReported(t *testing.T) {
	e := newEnv(t, testKey, func(context.Context, *llm.Request) (*llm.Response, error) {
		return nil, &llm.APIError{Provider: "fake", StatusCode: 401, Message: "bad key"}
	})
	done := doneOf(t, e.chat(e.createSession().ID, "你好"))
	if done.FinishReason != "error" || !strings.Contains(done.Error, "模型调用失败") || !strings.Contains(done.Error, "bad key") {
		t.Errorf("done = %+v", done)
	}
	if strings.Contains(done.Error, testKey) {
		t.Error("error leaks the key")
	}
}

func TestNewValidatesSettings(t *testing.T) {
	if _, err := New(nil, Settings{Provider: "x", MaxTurns: 1, MaxTokens: 1}, ""); err == nil {
		t.Error("invalid provider accepted")
	}
	s, err := New(nil, Settings{MaxTurns: 1, MaxTokens: 1}, "")
	if err != nil || s.settings.Provider != "chatcompletions" || s.settings.Root == "" {
		t.Errorf("defaults: %+v %v", s, err)
	}
}

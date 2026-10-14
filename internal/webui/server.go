// Package webui serves dio's local browser interface: a JSON/SSE API over the
// agent and session store, plus an embedded single-page front end.
package webui

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"

	"github.com/EvanFromDowntown/Dio/agent"
	"github.com/EvanFromDowntown/Dio/internal/cli"
	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/plugins/readfile"
	"github.com/EvanFromDowntown/Dio/session"
)

//go:embed index.html
var indexHTML []byte

// ClientFactory builds a model client; tests substitute a fake.
type ClientFactory func(provider, baseURL, apiKey string) (llm.Client, error)

type Server struct {
	store     session.Store
	newClient ClientFactory
	logger    *slog.Logger

	mu       sync.Mutex
	settings Settings
	apiKey   string
	running  map[string]context.CancelFunc
}

type Option func(*Server)

func WithClientFactory(f ClientFactory) Option { return func(s *Server) { s.newClient = f } }

func WithLogger(l *slog.Logger) Option { return func(s *Server) { s.logger = l } }

// New validates the initial settings. apiKey lives only in this process.
func New(store session.Store, settings Settings, apiKey string, opts ...Option) (*Server, error) {
	if err := settings.normalize(); err != nil {
		return nil, err
	}
	s := &Server{
		store: store, newClient: cli.NewProviderClient, logger: slog.New(slog.DiscardHandler),
		settings: settings, apiKey: apiKey, running: map[string]context.CancelFunc{},
	}
	for _, opt := range opts {
		opt(s)
	}
	return s, nil
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", s.index)
	mux.HandleFunc("GET /api/config", s.getConfig)
	mux.HandleFunc("PUT /api/config", s.putConfig)
	mux.HandleFunc("GET /api/sessions", s.listSessions)
	mux.HandleFunc("POST /api/sessions", s.createSession)
	mux.HandleFunc("GET /api/sessions/{id}", s.getSession)
	mux.HandleFunc("POST /api/sessions/{id}/messages", s.postMessage)
	mux.HandleFunc("POST /api/sessions/{id}/cancel", s.cancelRun)
	return localOnly(mux)
}

// localOnly rejects requests whose Host is not a loopback name (DNS
// rebinding) and cross-origin writes, so other web pages cannot drive the
// agent or read files through this server.
func localOnly(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !isLoopbackHost(r.Host) {
			writeError(w, http.StatusForbidden, "只接受来自本机的访问")
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			if origin := r.Header.Get("Origin"); origin != "" {
				u, err := url.Parse(origin)
				if err != nil || u.Host != r.Host {
					writeError(w, http.StatusForbidden, "拒绝跨站请求")
					return
				}
			}
			if !strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") {
				writeError(w, http.StatusUnsupportedMediaType, "请求体必须是 JSON")
				return
			}
		}
		w.Header().Set("X-Content-Type-Options", "nosniff")
		next.ServeHTTP(w, r)
	})
}

func isLoopbackHost(hostport string) bool {
	host := hostport
	if h, _, err := net.SplitHostPort(hostport); err == nil {
		host = h
	}
	host = strings.Trim(host, "[]")
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func (s *Server) index(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'")
	w.Write(indexHTML)
}

type configView struct {
	Settings
	HasAPIKey bool     `json:"has_api_key"`
	Providers []string `json:"providers"`
}

func (s *Server) config() configView {
	s.mu.Lock()
	defer s.mu.Unlock()
	return configView{Settings: s.settings, HasAPIKey: s.apiKey != "", Providers: cli.Providers}
}

func (s *Server) getConfig(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.config())
}

type configUpdate struct {
	Settings
	// APIKey replaces the key when non-empty; ClearAPIKey forgets it.
	APIKey      string `json:"api_key"`
	ClearAPIKey bool   `json:"clear_api_key"`
}

func (s *Server) putConfig(w http.ResponseWriter, r *http.Request) {
	var u configUpdate
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&u); err != nil {
		writeError(w, http.StatusBadRequest, "请求格式错误："+err.Error())
		return
	}
	if err := u.Settings.normalize(); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	s.mu.Lock()
	s.settings = u.Settings
	switch {
	case u.ClearAPIKey:
		s.apiKey = ""
	case strings.TrimSpace(u.APIKey) != "":
		s.apiKey = strings.TrimSpace(u.APIKey)
	}
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, s.config())
}

func (s *Server) listSessions(w http.ResponseWriter, r *http.Request) {
	infos, err := s.store.List(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, "读取会话失败："+err.Error())
		return
	}
	out := make([]SessionSummary, len(infos))
	for i, info := range infos {
		out[i] = summarize(info)
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) createSession(w http.ResponseWriter, r *http.Request) {
	cfg := s.config()
	sess, err := s.store.Create(r.Context(), session.Info{
		Provider: cfg.Provider, Model: cfg.Model,
		Metadata: map[string]string{"root": cfg.Root, "base_url": cfg.BaseURL, "client": "dio-ui"},
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, "创建会话失败："+err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, summarize(sess.Info))
}

type sessionView struct {
	SessionSummary
	Items   []Item `json:"items"`
	Running bool   `json:"running"`
}

func (s *Server) getSession(w http.ResponseWriter, r *http.Request) {
	sess, ok := s.loadSession(w, r)
	if !ok {
		return
	}
	s.mu.Lock()
	_, running := s.running[sess.ID]
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, sessionView{SessionSummary: summarize(sess.Info), Items: historyItems(sess.Messages), Running: running})
}

func (s *Server) loadSession(w http.ResponseWriter, r *http.Request) (*session.Session, bool) {
	sess, err := s.store.Get(r.Context(), r.PathValue("id"))
	switch {
	case errors.Is(err, session.ErrNotFound), errors.Is(err, session.ErrInvalidID):
		writeError(w, http.StatusNotFound, "会话不存在")
		return nil, false
	case err != nil:
		writeError(w, http.StatusInternalServerError, "读取会话失败："+err.Error())
		return nil, false
	}
	return sess, true
}

func (s *Server) cancelRun(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	cancel, ok := s.running[r.PathValue("id")]
	s.mu.Unlock()
	if ok {
		cancel()
	}
	writeJSON(w, http.StatusOK, map[string]bool{"canceled": ok})
}

type messageRequest struct {
	Text string `json:"text"`
}

// postMessage runs one user message and streams the result as SSE. Errors
// detected before the run starts are plain JSON responses.
func (s *Server) postMessage(w http.ResponseWriter, r *http.Request) {
	var req messageRequest
	body := http.MaxBytesReader(w, r.Body, 4<<20)
	if err := json.NewDecoder(body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "请求格式错误："+err.Error())
		return
	}
	// Reading to EOF lets net/http watch the connection, so the request
	// context (and the run) is canceled when the browser goes away.
	io.Copy(io.Discard, body)
	req.Text = strings.TrimSpace(req.Text)
	if req.Text == "" {
		writeError(w, http.StatusBadRequest, "消息不能为空")
		return
	}
	sess, ok := s.loadSession(w, r)
	if !ok {
		return
	}

	s.mu.Lock()
	cfg, key := s.settings, s.apiKey
	s.mu.Unlock()
	if key == "" {
		writeError(w, http.StatusBadRequest, "尚未设置 API key：请在“设置”中填写，或启动前设置环境变量")
		return
	}
	if cfg.Model == "" {
		writeError(w, http.StatusBadRequest, "尚未设置模型：请在“设置”中填写 model")
		return
	}
	client, err := s.newClient(cfg.Provider, cfg.BaseURL, key)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	run, err := s.newRun(r.Context(), cfg, client)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	defer run.close()

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	s.mu.Lock()
	if _, busy := s.running[sess.ID]; busy {
		s.mu.Unlock()
		writeError(w, http.StatusConflict, "该会话正在生成回答，请稍候或先停止")
		return
	}
	s.running[sess.ID] = cancel
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		delete(s.running, sess.ID)
		s.mu.Unlock()
	}()

	sse, err := newSSE(w)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	s.logger.Info("run started", "session", sess.ID, "provider", cfg.Provider, "model", cfg.Model)
	title := func() string {
		if latest, err := s.store.Get(context.WithoutCancel(ctx), sess.ID); err == nil {
			return latest.Title
		}
		return ""
	}
	streamRun(sse, cancel, run.runner.Stream(ctx, sess.ID, llm.UserText(req.Text)), sess.ID, title)
	s.logger.Info("run finished", "session", sess.ID)
}

type run struct {
	runner session.Runner
	files  *readfile.Plugin
}

func (r *run) close() { r.files.Close() }

func (s *Server) newRun(ctx context.Context, cfg Settings, client llm.Client) (*run, error) {
	ac := agent.Config{
		Client: client, Model: cfg.Model, System: cli.SystemPrompt,
		MaxTurns: cfg.MaxTurns, MaxTokens: cfg.MaxTokens, Temperature: cfg.Temperature,
	}
	if cfg.ReasoningEffort != "" {
		ac.Reasoning = &llm.ReasoningConfig{Effort: cfg.ReasoningEffort}
	}
	files := readfile.New(cfg.Root)
	ac.Plugins = []agent.Plugin{files, session.NewRecorder(s.store)}
	a, err := agent.New(ctx, ac)
	if err != nil {
		files.Close()
		return nil, fmt.Errorf("初始化失败：%w", err)
	}
	return &run{runner: session.Runner{Agent: a, Store: s.store}, files: files}, nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

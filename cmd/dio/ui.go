package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/EvanFromDowntown/Dio/internal/cli"
	"github.com/EvanFromDowntown/Dio/internal/webui"
	"github.com/EvanFromDowntown/Dio/session"
)

// serveUI runs the local web interface until interrupted.
func serveUI(opts *cli.Options, store *session.FileStore) int {
	host, _, err := net.SplitHostPort(opts.Addr)
	if err != nil {
		fmt.Fprintln(os.Stderr, "错误：-addr 格式应为 host:port：", err)
		return 2
	}
	if ip := net.ParseIP(host); host != "localhost" && (ip == nil || !ip.IsLoopback()) {
		fmt.Fprintf(os.Stderr, "错误：dio ui 只允许监听本机地址（127.0.0.1、::1 或 localhost），收到 %q\n", host)
		return 2
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	srv, err := webui.New(store, webui.SettingsFromOptions(opts), os.Getenv(opts.APIKeyEnv), webui.WithLogger(logger))
	if err != nil {
		fmt.Fprintln(os.Stderr, "错误：", err)
		return 2
	}
	ln, err := net.Listen("tcp", opts.Addr)
	if err != nil {
		fmt.Fprintln(os.Stderr, "错误：", err)
		return 1
	}
	httpSrv := &http.Server{Handler: srv.Handler(), ReadHeaderTimeout: 10 * time.Second}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		httpSrv.Shutdown(shutdownCtx)
	}()

	fmt.Fprintf(os.Stderr, "Dio Web 界面：http://%s/\n", ln.Addr())
	if os.Getenv(opts.APIKeyEnv) == "" {
		fmt.Fprintf(os.Stderr, "提示：环境变量 %s 未设置，请在界面“设置”中填写 API key。\n", opts.APIKeyEnv)
	}
	fmt.Fprintln(os.Stderr, "按 Ctrl+C 停止。")
	if err := httpSrv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
		fmt.Fprintln(os.Stderr, "错误：", err)
		return 1
	}
	return 0
}

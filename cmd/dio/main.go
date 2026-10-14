// Command dio is the interactive Dio client.
//
//	export DIO_API_KEY=...
//	dio -base-url https://gateway.example.com/v1 -model deepseek-flash
//	echo "总结 README.md" | dio -model deepseek-flash
//	dio -session 20261014-101500-a1b2c3
//	dio sessions
//	dio ui -addr 127.0.0.1:8080
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"

	"github.com/EvanFromDowntown/Dio/internal/cli"
	"github.com/EvanFromDowntown/Dio/session"
)

func main() {
	os.Exit(run())
}

func run() int {
	opts, err := cli.Parse(os.Args[1:], os.Getenv, os.Stderr)
	if errors.Is(err, cli.ErrHelp) {
		return 0
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "错误：", err)
		return 2
	}

	dir := opts.SessionsDir
	if dir == "" {
		if dir, err = session.DefaultDir(); err != nil {
			fmt.Fprintln(os.Stderr, "错误：", err)
			return 1
		}
	}
	store, err := session.NewFileStore(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "错误：", err)
		return 1
	}
	ctx := context.Background()

	if opts.Command == "sessions" {
		infos, err := store.List(ctx)
		if err == nil {
			err = cli.WriteSessions(os.Stdout, infos, "")
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, "错误：", err)
			return 1
		}
		return 0
	}

	if err := cli.ApplySessionDefaults(ctx, opts, store); err != nil {
		fmt.Fprintln(os.Stderr, "错误：", err)
		return 1
	}
	client, err := cli.NewClient(opts, os.Getenv)
	if err != nil {
		fmt.Fprintln(os.Stderr, "错误：", err)
		return 1
	}

	interactive := opts.Prompt == "" && isTerminal(os.Stdin)
	info := os.Stderr
	if interactive {
		info = os.Stdout
	}
	app, err := cli.NewApp(ctx, opts, client, store, os.Stdout, info)
	if err != nil {
		fmt.Fprintln(os.Stderr, "错误：", err)
		return 1
	}
	defer app.Close()

	if interactive {
		interrupts := make(chan os.Signal, 1)
		signal.Notify(interrupts, os.Interrupt)
		defer signal.Stop(interrupts)
		if err := app.REPL(ctx, os.Stdin, interrupts); err != nil {
			fmt.Fprintln(os.Stderr, "错误：", err)
			return 1
		}
		return 0
	}

	oneShotCtx, stop := signal.NotifyContext(ctx, os.Interrupt)
	defer stop()
	if err := app.OneShot(oneShotCtx, os.Stdin, opts.Prompt); err != nil {
		if oneShotCtx.Err() != nil {
			return 130
		}
		if app.SessionID() == "" {
			fmt.Fprintln(os.Stderr, "错误：", err)
		}
		return 1
	}
	return 0
}

func isTerminal(f *os.File) bool {
	info, err := f.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

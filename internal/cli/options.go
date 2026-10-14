// Package cli implements the interactive dio client: option parsing, slash
// commands, event rendering, the REPL and the one-shot (piped) mode.
package cli

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// Providers lists the supported -provider values.
var Providers = []string{"chatcompletions", "openai", "anthropic"}

const (
	DefaultProvider  = "chatcompletions"
	DefaultAPIKeyEnv = "DIO_API_KEY"
)

type Options struct {
	// Command is "sessions" for `dio sessions`, empty for chatting.
	Command string

	Provider  string
	Model     string
	BaseURL   string
	APIKeyEnv string

	Root        string
	SessionID   string
	SessionsDir string

	MaxTurns        int
	MaxTokens       int
	ReasoningEffort string
	Temperature     *float64
	ShowReasoning   bool

	// Prompt is the positional argument text; non-empty forces one-shot mode.
	Prompt string

	explicit map[string]bool
}

// Explicit reports whether name was set by a flag or environment variable.
func (o *Options) Explicit(name string) bool { return o.explicit[name] }

// ErrHelp is returned when -h/-help was requested; usage has been printed.
var ErrHelp = flag.ErrHelp

// envDefaults maps flags to the environment variables that provide defaults.
var envDefaults = map[string]string{
	"provider": "DIO_PROVIDER",
	"model":    "DIO_MODEL",
	"base-url": "DIO_BASE_URL",
}

// Parse parses command-line arguments (without the program name). Flags take
// precedence over DIO_PROVIDER, DIO_MODEL and DIO_BASE_URL.
func Parse(args []string, getenv func(string) string, usage io.Writer) (*Options, error) {
	o := &Options{explicit: map[string]bool{}}
	if len(args) > 0 && args[0] == "sessions" {
		o.Command = "sessions"
		args = args[1:]
	}

	fs := flag.NewFlagSet("dio", flag.ContinueOnError)
	fs.SetOutput(usage)
	fs.StringVar(&o.Provider, "provider", DefaultProvider, "协议："+strings.Join(Providers, " | ")+"（环境变量 DIO_PROVIDER）")
	fs.StringVar(&o.Model, "model", "", "模型名（环境变量 DIO_MODEL）")
	fs.StringVar(&o.BaseURL, "base-url", "", "API base URL（环境变量 DIO_BASE_URL）")
	fs.StringVar(&o.APIKeyEnv, "api-key-env", DefaultAPIKeyEnv, "存放 API key 的环境变量名")
	fs.StringVar(&o.Root, "root", ".", "read_file 可读取的工作目录（只读）")
	fs.StringVar(&o.SessionID, "session", "", "恢复指定会话")
	fs.StringVar(&o.SessionsDir, "sessions-dir", "", "会话目录（默认 ~/.dio/sessions）")
	fs.IntVar(&o.MaxTurns, "max-turns", 20, "每条消息最多的模型调用次数")
	fs.IntVar(&o.MaxTokens, "max-tokens", 4096, "每次模型调用的输出 token 上限（含推理）")
	fs.StringVar(&o.ReasoningEffort, "reasoning-effort", "", "推理强度：low | medium | high")
	fs.Func("temperature", "采样温度 0–2（默认由模型决定）", func(v string) error {
		t, err := strconv.ParseFloat(v, 64)
		if err != nil || t < 0 || t > 2 {
			return errors.New("必须是 0 到 2 之间的数字")
		}
		o.Temperature = &t
		return nil
	})
	fs.BoolVar(&o.ShowReasoning, "show-reasoning", false, "显示模型的推理过程")
	fs.Usage = func() {
		fmt.Fprint(usage, `用法：
  dio [参数]            交互模式（标准输入不是终端时，读取全部输入作为一条消息，回答后退出）
  dio [参数] "问题"      单次问答
  dio sessions          列出会话

参数：
`)
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	fs.Visit(func(f *flag.Flag) { o.explicit[f.Name] = true })
	for name, env := range envDefaults {
		if o.explicit[name] {
			continue
		}
		if v := getenv(env); v != "" {
			fs.Set(name, v)
			o.explicit[name] = true
		}
	}
	o.Prompt = strings.TrimSpace(strings.Join(fs.Args(), " "))

	if o.Command == "sessions" && o.Prompt != "" {
		return nil, fmt.Errorf("sessions 子命令不接受参数：%q", o.Prompt)
	}
	if !contains(Providers, o.Provider) {
		return nil, fmt.Errorf("未知的 provider %q，可选：%s", o.Provider, strings.Join(Providers, "、"))
	}
	switch o.ReasoningEffort {
	case "", "low", "medium", "high":
	default:
		return nil, fmt.Errorf("-reasoning-effort 只能是 low、medium 或 high，收到 %q", o.ReasoningEffort)
	}
	if o.MaxTurns <= 0 || o.MaxTokens <= 0 {
		return nil, errors.New("-max-turns 和 -max-tokens 必须大于 0")
	}
	return o, nil
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

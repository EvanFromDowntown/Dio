package webui

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/EvanFromDowntown/Dio/internal/cli"
)

// Settings are the user-adjustable run parameters. The API key is kept
// separately and never serialized.
type Settings struct {
	Provider        string   `json:"provider"`
	Model           string   `json:"model"`
	BaseURL         string   `json:"base_url"`
	Root            string   `json:"root"`
	MaxTurns        int      `json:"max_turns"`
	MaxTokens       int      `json:"max_tokens"`
	ReasoningEffort string   `json:"reasoning_effort"`
	Temperature     *float64 `json:"temperature"`
}

// SettingsFromOptions takes the initial settings from dio's flags.
func SettingsFromOptions(o *cli.Options) Settings {
	return Settings{
		Provider: o.Provider, Model: o.Model, BaseURL: o.BaseURL, Root: o.Root,
		MaxTurns: o.MaxTurns, MaxTokens: o.MaxTokens,
		ReasoningEffort: o.ReasoningEffort, Temperature: o.Temperature,
	}
}

// normalize validates s and makes Root absolute.
func (s *Settings) normalize() error {
	s.Provider = strings.TrimSpace(s.Provider)
	s.Model = strings.TrimSpace(s.Model)
	s.BaseURL = strings.TrimSpace(s.BaseURL)
	if s.Provider == "" {
		s.Provider = cli.DefaultProvider
	}
	if !slices.Contains(cli.Providers, s.Provider) {
		return fmt.Errorf("未知的 provider %q，可选：%s", s.Provider, strings.Join(cli.Providers, "、"))
	}
	switch s.ReasoningEffort {
	case "", "low", "medium", "high":
	default:
		return fmt.Errorf("推理强度只能是 low、medium 或 high，收到 %q", s.ReasoningEffort)
	}
	if t := s.Temperature; t != nil && (*t < 0 || *t > 2) {
		return errors.New("temperature 必须在 0 到 2 之间")
	}
	if s.MaxTurns <= 0 || s.MaxTokens <= 0 {
		return errors.New("最大轮数和最大 token 数必须大于 0")
	}
	root := s.Root
	if root == "" {
		root = "."
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return fmt.Errorf("工作目录无效：%w", err)
	}
	info, err := os.Stat(abs)
	if err != nil || !info.IsDir() {
		return fmt.Errorf("工作目录 %s 不存在或不是目录", abs)
	}
	s.Root = abs
	return nil
}

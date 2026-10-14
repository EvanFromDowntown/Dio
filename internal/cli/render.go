package cli

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/EvanFromDowntown/Dio/agent"
	"github.com/EvanFromDowntown/Dio/llm"
	"github.com/EvanFromDowntown/Dio/session"
)

// Renderer turns agent events into readable output. Answer text streams to
// Out; tool activity and reasoning go to Info, which may be the same writer.
type Renderer struct {
	out, info     *lineWriter
	showReasoning bool
	reasoning     map[int]bool
}

func NewRenderer(out, info io.Writer, showReasoning bool) *Renderer {
	r := &Renderer{out: &lineWriter{w: out}, showReasoning: showReasoning}
	r.info = r.out
	if info != out {
		r.info = &lineWriter{w: info}
	}
	return r
}

// lineWriter remembers whether the last write ended mid-line.
type lineWriter struct {
	w   io.Writer
	mid bool
}

func (l *lineWriter) write(s string) {
	if s == "" {
		return
	}
	io.WriteString(l.w, s)
	l.mid = !strings.HasSuffix(s, "\n")
}

func (l *lineWriter) endLine() {
	if l.mid {
		l.write("\n")
	}
}

func (r *Renderer) Event(ev agent.Event) {
	switch ev.Type {
	case agent.EventModel:
		r.modelEvent(ev.LLM)
	case agent.EventModelResponse:
		r.Finish()
	case agent.EventToolCall:
		r.Finish()
		r.info.write(fmt.Sprintf("[工具] %s(%s)\n", ev.ToolCall.Name, SummarizeArgs(ev.ToolCall.Arguments, 100)))
	case agent.EventToolResult:
		r.Finish()
		res := ev.ToolResult
		if res.IsError {
			r.info.write(fmt.Sprintf("[出错] %s：%s\n", ev.ToolCall.Name, Preview(res.Content, 120)))
			return
		}
		line := fmt.Sprintf("[结果] %s：%d 字节", ev.ToolCall.Name, len(res.Content))
		if p := Preview(res.Content, 60); p != "" {
			line += " · " + p
		}
		r.info.write(line + "\n")
	}
}

func (r *Renderer) modelEvent(ev *llm.StreamEvent) {
	switch ev.Type {
	case llm.EventStart:
		r.reasoning = nil
	case llm.EventTextDelta:
		if r.info != r.out {
			r.info.endLine()
		}
		r.out.write(ev.Delta)
	case llm.EventReasoningDelta:
		if !r.showReasoning || ev.Delta == "" {
			return
		}
		if r.reasoning == nil {
			r.reasoning = map[int]bool{}
		}
		if !r.reasoning[ev.Index] {
			r.reasoning[ev.Index] = true
			r.Finish()
			r.info.write("[思考] ")
		}
		r.info.write(ev.Delta)
	case llm.EventContentStop:
		if ev.Block != nil && ev.Block.Type == llm.ContentReasoning && r.reasoning[ev.Index] {
			r.info.endLine()
		}
	}
}

// Finish terminates partially written lines.
func (r *Renderer) Finish() {
	r.out.endLine()
	r.info.endLine()
}

// SummarizeArgs renders JSON object arguments as "k=v, …" within max runes,
// falling back to the raw text for anything else.
func SummarizeArgs(args json.RawMessage, max int) string {
	var obj map[string]any
	if err := json.Unmarshal(args, &obj); err != nil {
		return truncate(strings.Join(strings.Fields(string(args)), " "), max)
	}
	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, len(keys))
	for i, k := range keys {
		var v string
		switch val := obj[k].(type) {
		case string:
			v = val
		default:
			b, _ := json.Marshal(val)
			v = string(b)
		}
		parts[i] = k + "=" + truncate(strings.Join(strings.Fields(v), " "), 40)
	}
	return truncate(strings.Join(parts, ", "), max)
}

// Preview returns the first non-empty line of s, truncated.
func Preview(s string, max int) string {
	for _, line := range strings.Split(s, "\n") {
		if line = strings.TrimSpace(line); line != "" {
			return truncate(line, max)
		}
	}
	return ""
}

func truncate(s string, max int) string {
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max]) + "…"
}

// FormatUsage is the footer printed after each answer.
func FormatUsage(res *agent.Result) string {
	u := res.Usage
	var b bytes.Buffer
	fmt.Fprintf(&b, "（%d 次模型调用 · 输入 %d / 输出 %d tokens", res.Turns, u.InputTokens, u.OutputTokens)
	if u.CacheReadTokens > 0 {
		fmt.Fprintf(&b, " · 缓存命中 %d", u.CacheReadTokens)
	}
	if u.ReasoningTokens > 0 {
		fmt.Fprintf(&b, " · 推理 %d", u.ReasoningTokens)
	}
	b.WriteString("）")
	return b.String()
}

// WriteSessions prints a session table, most recent first.
func WriteSessions(w io.Writer, infos []session.Info, current string) error {
	if len(infos) == 0 {
		_, err := fmt.Fprintln(w, "还没有会话。")
		return err
	}
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "  ID\t更新时间\t消息数\tTokens（入/出）\t模型\t标题")
	for _, s := range infos {
		mark := " "
		if s.ID == current {
			mark = "*"
		}
		model := s.Model
		if s.Provider != "" {
			model = s.Provider + " · " + s.Model
		}
		fmt.Fprintf(tw, "%s %s\t%s\t%d\t%d/%d\t%s\t%s\n", mark, s.ID, s.UpdatedAt.Local().Format(time.DateTime),
			s.MessageCount, s.Usage.InputTokens, s.Usage.OutputTokens, model, truncate(s.Title, 40))
	}
	return tw.Flush()
}

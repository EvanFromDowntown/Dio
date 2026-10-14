package webui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"

	"github.com/EvanFromDowntown/Dio/agent"
	"github.com/EvanFromDowntown/Dio/llm"
)

type sseWriter struct {
	w http.ResponseWriter
	f http.Flusher
}

func newSSE(w http.ResponseWriter) (*sseWriter, error) {
	f, ok := w.(http.Flusher)
	if !ok {
		return nil, errors.New("当前连接不支持流式输出")
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream; charset=utf-8")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	f.Flush()
	return &sseWriter{w: w, f: f}, nil
}

// send writes one event; the data is the JSON encoding of v.
func (s *sseWriter) send(event string, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", event, data); err != nil {
		return err
	}
	s.f.Flush()
	return nil
}

// Stream event payloads. "item" events carry an Item (tool rows); text and
// reasoning arrive as deltas; "done" always ends the stream.
type deltaEvent struct {
	Delta string `json:"delta"`
}

type doneEvent struct {
	SessionID    string    `json:"session_id"`
	Title        string    `json:"title,omitempty"`
	FinishReason string    `json:"finish_reason"`
	Turns        int       `json:"turns"`
	Usage        llm.Usage `json:"usage"`
	Error        string    `json:"error,omitempty"`
}

// streamRun forwards a run's events; title is consulted once at the end so
// the page can show the title derived from the first message. A failed
// write means the browser is gone, so the run is canceled via cancel.
func streamRun(sse *sseWriter, cancel context.CancelFunc, events iter.Seq2[agent.Event, error], sessionID string, title func() string) {
	toolNames := map[string]string{}
	gone := false
	for ev, err := range events {
		if gone && ev.Type != agent.EventDone {
			continue
		}
		var werr error
		switch ev.Type {
		case agent.EventModel:
			switch ev.LLM.Type {
			case llm.EventTextDelta:
				if ev.LLM.Delta != "" {
					werr = sse.send("text", deltaEvent{Delta: ev.LLM.Delta})
				}
			case llm.EventReasoningDelta:
				if ev.LLM.Delta != "" {
					werr = sse.send("reasoning", deltaEvent{Delta: ev.LLM.Delta})
				}
			}
		case agent.EventModelResponse:
			werr = sse.send("message_end", struct{}{})
		case agent.EventToolCall:
			toolNames[ev.ToolCall.ID] = ev.ToolCall.Name
			werr = sse.send("item", toolCallItem(*ev.ToolCall))
		case agent.EventToolResult:
			werr = sse.send("item", toolResultItem(ev.ToolCall.Name, *ev.ToolResult))
		case agent.EventDone:
			done := doneEvent{SessionID: sessionID, Title: title()}
			if res := ev.Result; res != nil {
				done.FinishReason, done.Turns, done.Usage = string(res.FinishReason), res.Turns, res.Usage
			}
			if err != nil {
				done.Error = describe(err, done.FinishReason)
			}
			sse.send("done", done)
			return
		}
		if werr != nil && !gone {
			gone = true
			cancel()
		}
	}
}

func describe(err error, reason string) string {
	switch agent.FinishReason(reason) {
	case agent.FinishCanceled:
		return "已停止。已生成的内容和工具结果都已保存，可以继续发送消息。"
	case agent.FinishMaxTurns:
		return "已达到最大轮数限制。历史已保存，可以继续发送消息让模型接着做。"
	}
	var apiErr *llm.APIError
	if errors.As(err, &apiErr) {
		return "模型调用失败：" + apiErr.Error()
	}
	return err.Error()
}

// Package transport holds the HTTP and streaming plumbing shared by provider
// adapters.
package transport

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/EvanFromDowntown/Dio/internal/sse"
	"github.com/EvanFromDowntown/Dio/llm"
)

const maxErrorBody = 1 << 20

// PostJSON sends body to url. Non-2xx responses are consumed and returned as
// *llm.APIError; on success the caller owns resp.Body.
func PostJSON(ctx context.Context, hc *http.Client, provider, url string, header http.Header, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("%s: build request: %w", provider, err)
	}
	for k, vs := range header {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", provider, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		defer resp.Body.Close()
		data, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
		apiErr := ErrorFromBody(provider, data)
		apiErr.StatusCode = resp.StatusCode
		apiErr.RetryAfter = ParseRetryAfter(resp.Header.Get("Retry-After"), time.Now())
		return nil, apiErr
	}
	return resp, nil
}

// ErrorFromBody decodes the error envelopes used by both protocols:
// {"error":{"type","message","code"}} (optionally with a top-level "type")
// or a bare {"type","message","code"} object as sent in stream error events.
func ErrorFromBody(provider string, data []byte) *llm.APIError {
	apiErr := &llm.APIError{Provider: provider, Body: data}

	type detail struct {
		Type    string          `json:"type"`
		Message string          `json:"message"`
		Code    json.RawMessage `json:"code"`
	}
	var envelope struct {
		detail
		Error *detail `json:"error"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		apiErr.Message = truncate(strings.TrimSpace(string(data)), 512)
		return apiErr
	}
	d := envelope.detail
	if envelope.Error != nil {
		d = *envelope.Error
	}
	if d.Type != "error" {
		apiErr.Type = d.Type
	}
	apiErr.Message = d.Message
	apiErr.Code = rawToString(d.Code)
	if apiErr.Message == "" && apiErr.Type == "" && apiErr.Code == "" {
		apiErr.Message = truncate(strings.TrimSpace(string(data)), 512)
	}
	return apiErr
}

func rawToString(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	return string(raw)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// ParseRetryAfter accepts delta-seconds or an HTTP date.
func ParseRetryAfter(v string, now time.Time) time.Duration {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0
	}
	if secs, err := strconv.ParseFloat(v, 64); err == nil && secs > 0 {
		return time.Duration(secs * float64(time.Second))
	}
	if t, err := http.ParseTime(v); err == nil && t.After(now) {
		return t.Sub(now)
	}
	return 0
}

// MarshalWithExtra encodes v and merges extra into the top-level object.
func MarshalWithExtra(v any, extra map[string]any) ([]byte, error) {
	body, err := json.Marshal(v)
	if err != nil || len(extra) == 0 {
		return body, err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, err
	}
	for k, val := range extra {
		raw, err := json.Marshal(val)
		if err != nil {
			return nil, fmt.Errorf("encode extra field %q: %w", k, err)
		}
		fields[k] = raw
	}
	return json.Marshal(fields)
}

// DecodeFunc translates one SSE event into unified events. done marks the
// terminal event; a non-nil error aborts the stream.
type DecodeFunc func(ev sse.Event) (events []llm.StreamEvent, done bool, err error)

// NewStream adapts an SSE body to llm.Stream.
func NewStream(body io.ReadCloser, decode DecodeFunc) llm.Stream {
	return NewStreamWithEOF(body, decode, nil)
}

// EOFFunc is called when the body ends before a terminal event. It may
// complete the stream; returning done=false reports llm.ErrIncompleteStream.
type EOFFunc func() (events []llm.StreamEvent, done bool, err error)

func NewStreamWithEOF(body io.ReadCloser, decode DecodeFunc, onEOF EOFFunc) llm.Stream {
	return &stream{body: body, r: sse.NewReader(body), decode: decode, onEOF: onEOF}
}

type stream struct {
	body   io.ReadCloser
	r      *sse.Reader
	decode DecodeFunc
	onEOF  EOFFunc

	queue    []llm.StreamEvent
	cur      llm.StreamEvent
	err      error
	finished bool
	closed   bool
}

func (s *stream) Next() bool {
	for len(s.queue) == 0 {
		if s.finished || s.err != nil {
			s.Close()
			return false
		}
		var (
			events []llm.StreamEvent
			done   bool
		)
		ev, err := s.r.Next()
		switch {
		case errors.Is(err, io.EOF):
			err = nil
			if s.onEOF != nil {
				events, done, err = s.onEOF()
			}
			if err == nil && !done {
				err = llm.ErrIncompleteStream
			}
		case err == nil:
			events, done, err = s.decode(ev)
		}
		if err != nil {
			s.err = err
			continue
		}
		s.queue = append(s.queue, events...)
		s.finished = done
	}
	s.cur, s.queue = s.queue[0], s.queue[1:]
	return true
}

func (s *stream) Event() llm.StreamEvent { return s.cur }

func (s *stream) Err() error { return s.err }

func (s *stream) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true
	return s.body.Close()
}

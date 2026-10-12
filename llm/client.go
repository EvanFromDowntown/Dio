package llm

import (
	"context"
	"errors"
)

// Client is implemented by every provider adapter. Implementations must be
// safe for concurrent use.
type Client interface {
	// Generate performs a blocking request and returns the full response.
	Generate(ctx context.Context, req *Request) (*Response, error)
	// Stream starts a streaming request. HTTP-level failures are returned
	// here; failures after the stream started surface through Stream.Err.
	Stream(ctx context.Context, req *Request) (Stream, error)
}

// Stream is an iterator over streaming events:
//
//	s, err := client.Stream(ctx, req)
//	if err != nil { ... }
//	defer s.Close()
//	for s.Next() {
//		ev := s.Event()
//		...
//	}
//	if err := s.Err(); err != nil { ... }
//
// A successful stream always ends with an EventDone carrying the assembled
// Response.
type Stream interface {
	Next() bool
	Event() StreamEvent
	Err() error
	Close() error
}

type EventType string

const (
	// EventStart opens the response; Response carries ID and Model only.
	EventStart EventType = "start"
	// EventContentStart opens content block Index. Block holds its type and,
	// for tool calls, the call ID and name.
	EventContentStart EventType = "content_start"
	// EventTextDelta, EventReasoningDelta and EventToolCallDelta append Delta
	// to block Index. Tool call deltas are fragments of the JSON arguments.
	EventTextDelta      EventType = "text_delta"
	EventReasoningDelta EventType = "reasoning_delta"
	EventToolCallDelta  EventType = "tool_call_delta"
	// EventContentStop closes block Index; Block is the completed block.
	EventContentStop EventType = "content_stop"
	// EventDone is the final event; Response is the assembled response.
	EventDone EventType = "done"
)

// StreamEvent is a provider-agnostic streaming event. Index refers to the
// position of the block in the final Response.Content.
type StreamEvent struct {
	Type     EventType     `json:"type"`
	Index    int           `json:"index"`
	Delta    string        `json:"delta,omitempty"`
	Block    *ContentBlock `json:"block,omitempty"`
	Response *Response     `json:"response,omitempty"`
}

// ErrIncompleteStream is reported when a stream ends before EventDone.
var ErrIncompleteStream = errors.New("llm: stream ended before completion")

// Collect drains s, closes it, and returns the final response.
func Collect(s Stream) (*Response, error) {
	defer s.Close()
	var resp *Response
	for s.Next() {
		if ev := s.Event(); ev.Type == EventDone {
			resp = ev.Response
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	if resp == nil {
		return nil, ErrIncompleteStream
	}
	return resp, nil
}

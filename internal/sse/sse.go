// Package sse parses text/event-stream bodies.
package sse

import (
	"bufio"
	"errors"
	"io"
	"strings"
)

// Event is one dispatched server-sent event.
type Event struct {
	Name string
	Data string
	ID   string
}

// Reader reads events from an event stream. Lines are not length-limited.
type Reader struct {
	r *bufio.Reader
}

func NewReader(r io.Reader) *Reader {
	return &Reader{r: bufio.NewReader(r)}
}

// Next returns the next event, or io.EOF once the stream is exhausted.
// A trailing event not terminated by a blank line is still dispatched.
func (r *Reader) Next() (Event, error) {
	var (
		ev      Event
		data    []string
		hasData bool
	)
	for {
		line, err := r.r.ReadString('\n')
		if err != nil && !errors.Is(err, io.EOF) {
			return Event{}, err
		}
		eof := err != nil
		if eof && line == "" {
			if hasData || ev.Name != "" {
				ev.Data = strings.Join(data, "\n")
				return ev, nil
			}
			return Event{}, io.EOF
		}
		line = strings.TrimSuffix(line, "\n")
		line = strings.TrimSuffix(line, "\r")

		if line == "" {
			if hasData || ev.Name != "" {
				ev.Data = strings.Join(data, "\n")
				return ev, nil
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		field, value, _ := strings.Cut(line, ":")
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			ev.Name = value
		case "data":
			data = append(data, value)
			hasData = true
		case "id":
			ev.ID = value
		}
		if eof {
			if hasData || ev.Name != "" {
				ev.Data = strings.Join(data, "\n")
				return ev, nil
			}
			return Event{}, io.EOF
		}
	}
}

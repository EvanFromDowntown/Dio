package session

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/EvanFromDowntown/Dio/llm"
)

// formatVersion is written into each session header.
const formatVersion = 1

// record is one JSONL line. The first line of a session is a "session"
// header; every later line is an append-only change.
type record struct {
	Type    string       `json:"type"`
	At      time.Time    `json:"at"`
	Session *header      `json:"session,omitempty"`
	Message *llm.Message `json:"message,omitempty"`
	Usage   *llm.Usage   `json:"usage,omitempty"`
	Title   *string      `json:"title,omitempty"`
}

type header struct {
	Version  int               `json:"version"`
	ID       string            `json:"id"`
	Title    string            `json:"title,omitempty"`
	Provider string            `json:"provider,omitempty"`
	Model    string            `json:"model,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

const (
	recSession = "session"
	recMessage = "message"
	recUsage   = "usage"
	recTitle   = "title"
)

func headerRecord(info Info) record {
	return record{Type: recSession, At: info.CreatedAt, Session: &header{
		Version: formatVersion, ID: info.ID, Title: info.Title,
		Provider: info.Provider, Model: info.Model, Metadata: info.Metadata,
	}}
}

func messageRecords(at time.Time, msgs []llm.Message) []record {
	recs := make([]record, len(msgs))
	for i := range msgs {
		recs[i] = record{Type: recMessage, At: at, Message: &msgs[i]}
	}
	return recs
}

// encodeLines renders records as newline-terminated JSON lines.
func encodeLines(recs []record) ([]byte, error) {
	var buf bytes.Buffer
	for _, r := range recs {
		line, err := json.Marshal(r)
		if err != nil {
			return nil, fmt.Errorf("session: encode %s record: %w", r.Type, err)
		}
		buf.Write(line)
		buf.WriteByte('\n')
	}
	return buf.Bytes(), nil
}

// decodeSession replays JSONL data. A final line without its newline is a
// torn write from a crash and is ignored if it does not decode; any other
// undecodable line is an error.
func decodeSession(data []byte) (*Session, error) {
	torn := len(data) > 0 && data[len(data)-1] != '\n'
	lines := bytes.Split(data, []byte("\n"))
	var s *Session
	for i, line := range lines {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var r record
		if err := json.Unmarshal(line, &r); err != nil {
			if torn && i == len(lines)-1 {
				break
			}
			return nil, fmt.Errorf("session: corrupt record on line %d: %w", i+1, err)
		}
		if s == nil {
			if r.Type != recSession || r.Session == nil {
				return nil, errors.New("session: missing session header")
			}
			h := r.Session
			s = &Session{Info: Info{
				ID: h.ID, Title: h.Title, Provider: h.Provider, Model: h.Model, Metadata: h.Metadata,
				CreatedAt: r.At, UpdatedAt: r.At,
			}, Messages: []llm.Message{}}
			continue
		}
		if err := s.apply(r); err != nil {
			return nil, fmt.Errorf("session: line %d: %w", i+1, err)
		}
	}
	if s == nil {
		return nil, errors.New("session: missing session header")
	}
	return s, nil
}

func (s *Session) apply(r record) error {
	switch r.Type {
	case recMessage:
		if r.Message == nil {
			return errors.New("message record without message")
		}
		s.Messages = append(s.Messages, *r.Message)
		s.MessageCount = len(s.Messages)
	case recUsage:
		if r.Usage != nil {
			addUsage(&s.Usage, *r.Usage)
		}
	case recTitle:
		if r.Title != nil {
			s.Title = *r.Title
		}
	default:
		// Unknown record types from newer versions are skipped.
		return nil
	}
	if r.At.After(s.UpdatedAt) {
		s.UpdatedAt = r.At
	}
	return nil
}

// Package session persists conversations (one or more agent runs) so they can
// be listed and resumed.
package session

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"regexp"
	"time"

	"github.com/EvanFromDowntown/Dio/llm"
)

var (
	ErrNotFound  = errors.New("session: not found")
	ErrExists    = errors.New("session: already exists")
	ErrInvalidID = errors.New("session: invalid id")
)

// Info is session metadata, without the message history.
type Info struct {
	ID        string    `json:"id"`
	Title     string    `json:"title,omitempty"`
	Provider  string    `json:"provider,omitempty"`
	Model     string    `json:"model,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	// Metadata holds caller-defined string attributes set at creation.
	Metadata     map[string]string `json:"metadata,omitempty"`
	MessageCount int               `json:"message_count"`
	// Usage accumulates every model response recorded in the session.
	Usage llm.Usage `json:"usage"`
}

type Session struct {
	Info
	Messages []llm.Message `json:"messages"`
}

// Store persists sessions. Implementations must be safe for concurrent use.
type Store interface {
	// Create starts a session. An empty info.ID gets a generated one;
	// CreatedAt and UpdatedAt are set by the store.
	Create(ctx context.Context, info Info) (*Session, error)
	Get(ctx context.Context, id string) (*Session, error)
	// List returns all sessions, most recently updated first.
	List(ctx context.Context) ([]Info, error)
	AppendMessages(ctx context.Context, id string, msgs ...llm.Message) error
	AddUsage(ctx context.Context, id string, usage llm.Usage) error
	SetTitle(ctx context.Context, id string, title string) error
	Delete(ctx context.Context, id string) error
}

var idPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$`)

// ValidateID rejects IDs that are unsafe to use as file names.
func ValidateID(id string) error {
	if !idPattern.MatchString(id) {
		return fmt.Errorf("%w: %q", ErrInvalidID, id)
	}
	return nil
}

// NewID returns a time-sortable random ID like "20261012-171800-a1b2c3".
func NewID(now time.Time) string {
	var b [3]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(err)
	}
	return now.UTC().Format("20060102-150405") + "-" + hex.EncodeToString(b[:])
}

// PendingToolCalls returns the tool calls in the final assistant message that
// have no result yet. A history only ends this way if the process died
// between recording a response and its tool results.
func PendingToolCalls(msgs []llm.Message) []llm.ToolCall {
	if len(msgs) == 0 || msgs[len(msgs)-1].Role != llm.RoleAssistant {
		return nil
	}
	return msgs[len(msgs)-1].ToolCalls()
}

// Title derives a short title from the first user text in msgs.
func Title(msgs []llm.Message) string {
	for _, m := range msgs {
		if m.Role != llm.RoleUser {
			continue
		}
		if t := []rune(m.Text()); len(t) > 0 {
			if len(t) > 60 {
				return string(t[:60]) + "…"
			}
			return string(t)
		}
	}
	return ""
}

type ctxKey struct{}

// NewContext attaches a session ID; Recorder persists into that session.
func NewContext(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, ctxKey{}, id)
}

func IDFromContext(ctx context.Context) (string, bool) {
	id, ok := ctx.Value(ctxKey{}).(string)
	return id, ok && id != ""
}

func addUsage(dst *llm.Usage, u llm.Usage) {
	dst.InputTokens += u.InputTokens
	dst.OutputTokens += u.OutputTokens
	dst.CacheReadTokens += u.CacheReadTokens
	dst.CacheWriteTokens += u.CacheWriteTokens
	dst.ReasoningTokens += u.ReasoningTokens
}

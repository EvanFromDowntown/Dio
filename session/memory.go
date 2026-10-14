package session

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/EvanFromDowntown/Dio/llm"
)

// MemoryStore keeps sessions in memory. It stores the same encoded records as
// FileStore, so callers never share memory with the store and serialization
// is exercised exactly as on disk.
type MemoryStore struct {
	mu   sync.Mutex
	data map[string][]byte
	now  func() time.Time
}

var _ Store = (*MemoryStore)(nil)

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{data: map[string][]byte{}, now: time.Now}
}

func (s *MemoryStore) Create(_ context.Context, info Info) (*Session, error) {
	now := s.now()
	if info.ID == "" {
		info.ID = NewID(now)
	}
	if err := ValidateID(info.ID); err != nil {
		return nil, err
	}
	info.CreatedAt, info.UpdatedAt = now, now
	info.Provider = llm.NormalizeProvider(info.Provider)
	info.MessageCount, info.Usage = 0, llm.Usage{}
	line, err := encodeLines([]record{headerRecord(info)})
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.data[info.ID]; ok {
		return nil, fmt.Errorf("%w: %s", ErrExists, info.ID)
	}
	s.data[info.ID] = line
	return &Session{Info: info, Messages: []llm.Message{}}, nil
}

func (s *MemoryStore) Get(_ context.Context, id string) (*Session, error) {
	s.mu.Lock()
	data, ok := s.data[id]
	s.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrNotFound, id)
	}
	return decodeSession(data)
}

func (s *MemoryStore) List(ctx context.Context) ([]Info, error) {
	s.mu.Lock()
	ids := make([]string, 0, len(s.data))
	for id := range s.data {
		ids = append(ids, id)
	}
	s.mu.Unlock()
	var out []Info
	for _, id := range ids {
		if sess, err := s.Get(ctx, id); err == nil {
			out = append(out, sess.Info)
		}
	}
	sortInfos(out)
	return out, nil
}

func (s *MemoryStore) AppendMessages(_ context.Context, id string, msgs ...llm.Message) error {
	if len(msgs) == 0 {
		return nil
	}
	return s.append(id, messageRecords(s.now(), msgs))
}

func (s *MemoryStore) AddUsage(_ context.Context, id string, usage llm.Usage) error {
	return s.append(id, []record{{Type: recUsage, At: s.now(), Usage: &usage}})
}

func (s *MemoryStore) SetTitle(_ context.Context, id, title string) error {
	return s.append(id, []record{{Type: recTitle, At: s.now(), Title: &title}})
}

func (s *MemoryStore) Delete(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.data[id]; !ok {
		return fmt.Errorf("%w: %s", ErrNotFound, id)
	}
	delete(s.data, id)
	return nil
}

func (s *MemoryStore) append(id string, recs []record) error {
	lines, err := encodeLines(recs)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	data, ok := s.data[id]
	if !ok {
		return fmt.Errorf("%w: %s", ErrNotFound, id)
	}
	s.data[id] = append(data, lines...)
	return nil
}

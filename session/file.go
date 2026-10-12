package session

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/EvanFromDowntown/Dio/llm"
)

// FileStore keeps one append-only JSONL file per session in a directory.
// Each append is a single write of complete lines, so a crash can at worst
// leave a torn final line, which readers ignore and the next append removes.
// Concurrency is coordinated within one process; separate processes writing
// the same session are not supported.
type FileStore struct {
	dir  string
	sync bool
	now  func() time.Time
	mu   sync.Mutex
}

var _ Store = (*FileStore)(nil)

type FileOption func(*FileStore)

// WithFsync makes every write call fsync before returning.
func WithFsync(enabled bool) FileOption { return func(s *FileStore) { s.sync = enabled } }

// WithClock overrides time.Now (for tests).
func WithClock(now func() time.Time) FileOption { return func(s *FileStore) { s.now = now } }

// DefaultDir is ~/.dio/sessions.
func DefaultDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".dio", "sessions"), nil
}

// NewFileStore creates dir (mode 0700) if needed.
func NewFileStore(dir string, opts ...FileOption) (*FileStore, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("session: %w", err)
	}
	s := &FileStore{dir: dir, now: time.Now}
	for _, opt := range opts {
		opt(s)
	}
	return s, nil
}

func (s *FileStore) Dir() string { return s.dir }

func (s *FileStore) path(id string) string { return filepath.Join(s.dir, id+".jsonl") }

func (s *FileStore) Create(_ context.Context, info Info) (*Session, error) {
	now := s.now()
	if info.ID == "" {
		info.ID = NewID(now)
	}
	if err := ValidateID(info.ID); err != nil {
		return nil, err
	}
	info.CreatedAt, info.UpdatedAt = now, now
	info.MessageCount, info.Usage = 0, llm.Usage{}
	data, err := encodeLines([]record{headerRecord(info)})
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	f, err := os.OpenFile(s.path(info.ID), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if errors.Is(err, fs.ErrExist) {
		return nil, fmt.Errorf("%w: %s", ErrExists, info.ID)
	}
	if err != nil {
		return nil, fmt.Errorf("session: %w", err)
	}
	defer f.Close()
	if err := s.write(f, data); err != nil {
		return nil, err
	}
	return &Session{Info: info, Messages: []llm.Message{}}, nil
}

func (s *FileStore) Get(_ context.Context, id string) (*Session, error) {
	if err := ValidateID(id); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.load(id)
}

func (s *FileStore) load(id string) (*Session, error) {
	data, err := os.ReadFile(s.path(id))
	if errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("%w: %s", ErrNotFound, id)
	}
	if err != nil {
		return nil, fmt.Errorf("session: %w", err)
	}
	sess, err := decodeSession(data)
	if err != nil {
		return nil, fmt.Errorf("%w (%s)", err, id)
	}
	return sess, nil
}

// List skips files that cannot be decoded.
func (s *FileStore) List(_ context.Context) ([]Info, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, fmt.Errorf("session: %w", err)
	}
	var out []Info
	for _, e := range entries {
		id, ok := strings.CutSuffix(e.Name(), ".jsonl")
		if !ok || e.IsDir() || ValidateID(id) != nil {
			continue
		}
		sess, err := s.load(id)
		if err != nil {
			continue
		}
		out = append(out, sess.Info)
	}
	sortInfos(out)
	return out, nil
}

func (s *FileStore) AppendMessages(_ context.Context, id string, msgs ...llm.Message) error {
	if len(msgs) == 0 {
		return nil
	}
	return s.append(id, messageRecords(s.now(), msgs))
}

func (s *FileStore) AddUsage(_ context.Context, id string, usage llm.Usage) error {
	return s.append(id, []record{{Type: recUsage, At: s.now(), Usage: &usage}})
}

func (s *FileStore) SetTitle(_ context.Context, id, title string) error {
	return s.append(id, []record{{Type: recTitle, At: s.now(), Title: &title}})
}

func (s *FileStore) Delete(_ context.Context, id string) error {
	if err := ValidateID(id); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	err := os.Remove(s.path(id))
	if errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("%w: %s", ErrNotFound, id)
	}
	return err
}

func (s *FileStore) append(id string, recs []record) error {
	if err := ValidateID(id); err != nil {
		return err
	}
	data, err := encodeLines(recs)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	f, err := os.OpenFile(s.path(id), os.O_RDWR|os.O_APPEND, 0)
	if errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("%w: %s", ErrNotFound, id)
	}
	if err != nil {
		return fmt.Errorf("session: %w", err)
	}
	defer f.Close()
	if err := dropTornTail(f); err != nil {
		return fmt.Errorf("session: repair %s: %w", id, err)
	}
	return s.write(f, data)
}

func (s *FileStore) write(f *os.File, data []byte) error {
	if _, err := f.Write(data); err != nil {
		return fmt.Errorf("session: write: %w", err)
	}
	if s.sync {
		if err := f.Sync(); err != nil {
			return fmt.Errorf("session: fsync: %w", err)
		}
	}
	return nil
}

// dropTornTail truncates a final line that lacks its newline, i.e. a write
// that never completed, so new records start on a fresh line.
func dropTornTail(f *os.File) error {
	info, err := f.Stat()
	if err != nil {
		return err
	}
	size := info.Size()
	if size == 0 {
		return nil
	}
	last := make([]byte, 1)
	if _, err := f.ReadAt(last, size-1); err != nil {
		return err
	}
	if last[0] == '\n' {
		return nil
	}
	const chunk = 64 << 10
	buf := make([]byte, chunk)
	for end := size; end > 0; {
		start := max(end-chunk, 0)
		n, err := f.ReadAt(buf[:end-start], start)
		if err != nil && !errors.Is(err, io.EOF) {
			return err
		}
		for i := n - 1; i >= 0; i-- {
			if buf[i] == '\n' {
				return f.Truncate(start + int64(i) + 1)
			}
		}
		end = start
	}
	return f.Truncate(0)
}

func sortInfos(infos []Info) {
	sort.SliceStable(infos, func(i, j int) bool {
		if !infos[i].UpdatedAt.Equal(infos[j].UpdatedAt) {
			return infos[i].UpdatedAt.After(infos[j].UpdatedAt)
		}
		return infos[i].ID > infos[j].ID
	})
}

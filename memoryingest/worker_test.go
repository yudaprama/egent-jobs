package memoryingest

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"strings"
	"sync"
	"testing"
)

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// fakeStore records every Create* call so tests can assert on
// the extracted memories without a real Postgres.
type fakeStore struct {
	mu          sync.Mutex
	identities  []CreateIdentityInput
	activities  []CreateActivityInput
	contexts    []CreateContextInput
	experiences []CreateExperienceInput
	preferences []CreatePreferenceInput
	failNext    bool
}

var errFake = stringErr("fake store error")

type stringErr string

func (e stringErr) Error() string { return string(e) }

func (f *fakeStore) CreateIdentity(_ context.Context, _ string, in CreateIdentityInput) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failNext {
		f.failNext = false
		return "", errFake
	}
	f.identities = append(f.identities, in)
	return "fake-idn", nil
}
func (f *fakeStore) CreateActivity(_ context.Context, _ string, in CreateActivityInput) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.activities = append(f.activities, in)
	return "fake-act", nil
}
func (f *fakeStore) CreateContext(_ context.Context, _ string, in CreateContextInput) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.contexts = append(f.contexts, in)
	return "fake-ctx", nil
}
func (f *fakeStore) CreateExperience(_ context.Context, _ string, in CreateExperienceInput) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.experiences = append(f.experiences, in)
	return "fake-exp", nil
}
func (f *fakeStore) CreatePreference(_ context.Context, _ string, in CreatePreferenceInput) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.preferences = append(f.preferences, in)
	return "fake-prf", nil
}

// scriptedLLM returns canned replies in order. Each call consumes the
// next entry; the last entry is repeated for any further calls.
type scriptedLLM struct {
	replies []string
	calls   int
	mu      sync.Mutex
}

func (s *scriptedLLM) Chat(_ context.Context, _, _ string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.replies) == 0 {
		return "[]", nil
	}
	idx := s.calls
	if idx >= len(s.replies) {
		idx = len(s.replies) - 1
	}
	s.calls++
	return s.replies[idx], nil
}

func TestWorker_GatekeeperApprovesThenPersists(t *testing.T) {
	store := &fakeStore{}
	llm := &scriptedLLM{replies: []string{
		`{"relevant": true, "reason": "personal"}`,
		`[{"description":"Alice is a researcher","type":"professional","role":"scientist"}]`,
		`[{"type":"running","narrative":"5k this morning"}]`,
		`[{"title":"Q3 launch","description":"shipping a v2"}]`,
		`[{"situation":"stuck on bug","action":"reverted the PR"}]`,
		`[{"suggestions":"prefers dark mode"}]`,
	}}
	w := NewIngestWorker(Config{Store: store, LLM: llm, Logger: discardLogger()})

	if err := w.run(context.Background(), MemoryIngestArgs{
		UserID: "u-1", TopicID: "t-1", Content: "Alice is a researcher who runs every morning.",
	}, discardLogger()); err != nil {
		t.Fatalf("run: %v", err)
	}
	if got := len(store.identities); got != 1 {
		t.Errorf("expected 1 identity, got %d", got)
	}
	if got := len(store.activities); got != 1 {
		t.Errorf("expected 1 activity, got %d", got)
	}
	if got := len(store.contexts); got != 1 {
		t.Errorf("expected 1 context, got %d", got)
	}
	if got := len(store.experiences); got != 1 {
		t.Errorf("expected 1 experience, got %d", got)
	}
	if got := len(store.preferences); got != 1 {
		t.Errorf("expected 1 preference, got %d", got)
	}
}

func TestWorker_GatekeeperRejectsSkipsExtraction(t *testing.T) {
	store := &fakeStore{}
	llm := &scriptedLLM{replies: []string{
		`{"relevant": false, "reason": "small talk"}`,
	}}
	w := NewIngestWorker(Config{Store: store, LLM: llm, Logger: discardLogger()})

	if err := w.run(context.Background(), MemoryIngestArgs{
		UserID: "u-1", Content: "Hi, how are you?",
	}, discardLogger()); err != nil {
		t.Fatalf("run: %v", err)
	}
	if llm.calls != 1 {
		t.Errorf("expected only gatekeeper call, got %d", llm.calls)
	}
	if len(store.identities) != 0 || len(store.activities) != 0 || len(store.preferences) != 0 {
		t.Error("expected no memories to be persisted")
	}
}

func TestWorker_EmptyContentNoop(t *testing.T) {
	store := &fakeStore{}
	llm := &scriptedLLM{}
	w := NewIngestWorker(Config{Store: store, LLM: llm, Logger: discardLogger()})

	if err := w.run(context.Background(), MemoryIngestArgs{UserID: "u-1"}, discardLogger()); err != nil {
		t.Fatalf("run: %v", err)
	}
	if llm.calls != 0 {
		t.Errorf("expected no LLM calls, got %d", llm.calls)
	}
}

func TestWorker_MalformedReplyDefaultsToExtract(t *testing.T) {
	store := &fakeStore{}
	llm := &scriptedLLM{replies: []string{
		"not json at all",
		`[]`, `[]`, `[]`, `[]`, `[]`,
	}}
	w := NewIngestWorker(Config{Store: store, LLM: llm, Logger: discardLogger()})

	if err := w.run(context.Background(), MemoryIngestArgs{UserID: "u-1", Content: "anything"}, discardLogger()); err != nil {
		t.Fatalf("run: %v", err)
	}
	if llm.calls != 6 {
		t.Errorf("expected 6 LLM calls, got %d", llm.calls)
	}
}

func TestWorker_PersistFailureDoesNotAbort(t *testing.T) {
	store := &fakeStore{failNext: true}
	llm := &scriptedLLM{replies: []string{
		`{"relevant": true}`,
		`[{"description":"x","type":"personal"}]`,
		`[]`, `[]`, `[]`, `[]`,
	}}
	w := NewIngestWorker(Config{Store: store, LLM: llm, Logger: discardLogger()})

	if err := w.run(context.Background(), MemoryIngestArgs{UserID: "u-1", Content: "x"}, discardLogger()); err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(store.identities) != 0 {
		t.Error("identity should not have been persisted")
	}
	if llm.calls != 6 {
		t.Errorf("expected all 6 calls, got %d", llm.calls)
	}
}

func TestGatekeeperJSONLine(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"plain", `{"relevant":true,"reason":"x"}`, `{"relevant":true,"reason":"x"}`},
		{"wrapped in prose", `Sure! {"relevant":false,"reason":"y"}`, `{"relevant":false,"reason":"y"}`},
		{"missing", `not json`, ``},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := gatekeeperJSONLine.FindString(c.in)
			if got != c.want {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}

func TestParseArrayReply(t *testing.T) {
	type item struct {
		Description string `json:"description"`
	}
	cases := []struct {
		name string
		in   string
		want int
	}{
		{"plain", `[{"description":"a"},{"description":"b"}]`, 2},
		{"empty", `[]`, 0},
		{"wrapped", "let me extract:\n[{\"description\":\"a\"}]\nok", 1},
		{"no array", `just prose`, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := parseArrayReply[item](c.in)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			if len(got) != c.want {
				t.Errorf("got %d items, want %d", len(got), c.want)
			}
		})
	}
}

func TestExtractIdentityFiltersEmpty(t *testing.T) {
	llm := &scriptedLLM{replies: []string{
		`[{"description":"real","type":"personal"},{"description":""},{"type":"professional"}]`,
	}}
	w := NewIngestWorker(Config{Store: &fakeStore{}, LLM: llm, Logger: discardLogger()})
	got, err := w.extractIdentity(context.Background(), "x", discardLogger())
	if err != nil {
		t.Fatalf("extractIdentity: %v", err)
	}
	if len(got) != 1 {
		t.Errorf("expected 1 valid identity (others dropped), got %d", len(got))
	}
	if got[0].Description != "real" {
		t.Errorf("expected description 'real', got %q", got[0].Description)
	}
}

func TestTruncate(t *testing.T) {
	if got := truncate("hello world", 5); got != "hello…" {
		t.Errorf("truncate = %q", got)
	}
	if got := truncate("hi", 10); got != "hi" {
		t.Errorf("truncate short = %q", got)
	}
	if got := strings.HasPrefix("hello", "h"); !got {
		t.Error("strings.HasPrefix import check")
	}
}

var (
	_ Store     = (*fakeStore)(nil)
	_ LLMClient = (*scriptedLLM)(nil)
	_ json.Marshaler
)
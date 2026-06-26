package memoryingest

import (
	"context"
	"sync"
	"testing"
	"time"
)

// fakePersonaStore is an in-memory PersonaStore used by the persona
// worker tests.
type fakePersonaStore struct {
	mu        sync.Mutex
	memories  []MemoryRow
	latest    *PersonaDoc
	written   []PersonaDoc
	failRead  bool
}

func (f *fakePersonaStore) ReadRecentMemories(_ context.Context, _ string, _ time.Time, _ int) ([]MemoryRow, error) {
	if f.failRead {
		return nil, errFake
	}
	return f.memories, nil
}
func (f *fakePersonaStore) ReadLatestPersona(_ context.Context, _ string) (*PersonaDoc, error) {
	return f.latest, nil
}
func (f *fakePersonaStore) WritePersona(_ context.Context, userID, persona, tagline string, memoryIDs []string) (*PersonaDoc, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	p := PersonaDoc{
		ID: "fake-id", UserID: userID, Profile: "default",
		Tagline: tagline, Persona: persona, Version: 1,
		MemoryIDs: memoryIDs,
	}
	f.written = append(f.written, p)
	return &p, nil
}

func TestPersonaWorker_PersistsFreshPersona(t *testing.T) {
	store := &fakePersonaStore{memories: []MemoryRow{
		{ID: "m1", Layer: "identity", Summary: "Alice is a researcher"},
		{ID: "m2", Layer: "preference", Summary: "prefers dark mode"},
	}}
	llm := &scriptedLLM{replies: []string{
		`{"tagline":"Researcher who likes dark mode","persona":"Alice is a researcher..."}`,
	}}
	w := NewPersonaWorker(PersonaConfig{Store: store, LLM: llm, Logger: discardLogger()})

	if err := w.run(context.Background(), PersonaRefreshArgs{UserID: "u-1"}, discardLogger()); err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(store.written) != 1 {
		t.Fatalf("expected 1 write, got %d", len(store.written))
	}
	if store.written[0].Persona == "" {
		t.Error("expected non-empty persona")
	}
}

func TestPersonaWorker_SkipsWhenNoMemories(t *testing.T) {
	store := &fakePersonaStore{memories: nil}
	llm := &scriptedLLM{}
	w := NewPersonaWorker(PersonaConfig{Store: store, LLM: llm, Logger: discardLogger()})

	if err := w.run(context.Background(), PersonaRefreshArgs{UserID: "u-1"}, discardLogger()); err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(store.written) != 0 {
		t.Errorf("expected no writes, got %d", len(store.written))
	}
	if llm.calls != 0 {
		t.Errorf("expected no LLM calls, got %d", llm.calls)
	}
}

func TestPersonaWorker_SkipsEmptyPersona(t *testing.T) {
	store := &fakePersonaStore{memories: []MemoryRow{{ID: "m1", Layer: "identity", Summary: "x"}}}
	llm := &scriptedLLM{replies: []string{`{"tagline":"","persona":""}`}}
	w := NewPersonaWorker(PersonaConfig{Store: store, LLM: llm, Logger: discardLogger()})

	if err := w.run(context.Background(), PersonaRefreshArgs{UserID: "u-1"}, discardLogger()); err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(store.written) != 0 {
		t.Errorf("expected no writes for empty persona, got %d", len(store.written))
	}
}

func TestParsePersonaReply_Valid(t *testing.T) {
	tagline, persona := parsePersonaReply(`sure! {"tagline":"T","persona":"P"}`, nil)
	if tagline != "T" {
		t.Errorf("tagline = %q", tagline)
	}
	if persona != "P" {
		t.Errorf("persona = %q", persona)
	}
}

func TestParsePersonaReply_MalformedFallsBack(t *testing.T) {
	prev := &PersonaDoc{Tagline: "OLD", Persona: "OLD-P"}
	tagline, persona := parsePersonaReply("not json at all", prev)
	if tagline != "OLD" {
		t.Errorf("fallback tagline = %q", tagline)
	}
	if persona == "" {
		t.Error("fallback persona should be the raw reply")
	}
}

func TestBuildPersonaPrompts_IncludesPrevious(t *testing.T) {
	sys, usr := buildPersonaPrompts(&PersonaDoc{
		Version: 3, Tagline: "T3", Persona: "P3",
		UpdatedAt: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
	}, []MemoryRow{{ID: "m1", Layer: "identity", Summary: "x"}})
	if sys == "" || usr == "" {
		t.Error("expected non-empty prompts")
	}
	if !contains(usr, "Previous persona") {
		t.Error("user prompt should include previous persona")
	}
	if !contains(usr, "v3") {
		t.Error("user prompt should include version")
	}
	if !contains(usr, "New memories") {
		t.Error("user prompt should include new memories header")
	}
}

func TestBuildPersonaPrompts_NoPrevious(t *testing.T) {
	_, usr := buildPersonaPrompts(nil, []MemoryRow{{ID: "m1", Layer: "identity"}})
	if !contains(usr, "(none yet)") {
		t.Error("expected '(none yet)' when no previous persona")
	}
}

func TestPersonaWorker_NilStoreReturnsError(t *testing.T) {
	w := NewPersonaWorker(PersonaConfig{Store: nil, LLM: &scriptedLLM{}, Logger: discardLogger()})
	if err := w.run(context.Background(), PersonaRefreshArgs{UserID: "u-1"}, discardLogger()); err == nil {
		t.Error("expected error when store is nil")
	}
}

func TestPersonaWorker_NilLLMReturnsError(t *testing.T) {
	w := NewPersonaWorker(PersonaConfig{Store: &fakePersonaStore{}, LLM: nil, Logger: discardLogger()})
	if err := w.run(context.Background(), PersonaRefreshArgs{UserID: "u-1"}, discardLogger()); err == nil {
		t.Error("expected error when LLM is nil")
	}
}

func TestDiffOrEmpty(t *testing.T) {
	if got := diffOrEmpty("a", "a"); got != "" {
		t.Errorf("expected empty for same, got %q", got)
	}
	if got := diffOrEmpty("a", "b"); got != "b" {
		t.Errorf("expected new value, got %q", got)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && indexOf(s, substr) >= 0
}

func indexOf(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		match := true
		for j := 0; j < len(substr); j++ {
			if s[i+j] != substr[j] {
				match = false
				break
			}
		}
		if match {
			return i
		}
	}
	return -1
}

var _ PersonaStore = (*fakePersonaStore)(nil)
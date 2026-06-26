package memoryingest

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"

	"egent-jobs/embeddings"
)

// stubEmbedder returns a fixed vector for every input, so embed() can be
// exercised without a network call.
type stubEmbedder struct {
	vec []float32
	err error
}

func (s stubEmbedder) Model() string { return "stub" }
func (s stubEmbedder) EmbedBatch(_ context.Context, inputs []embeddings.Input) ([]embeddings.Result, error) {
	if s.err != nil {
		return nil, s.err
	}
	out := make([]embeddings.Result, len(inputs))
	for i, in := range inputs {
		out[i] = embeddings.Result{ChunkID: in.ChunkID, Vector: s.vec}
	}
	return out, nil
}

func TestVecToString(t *testing.T) {
	if got := vecToString([]float32{1, 2.5, -3}); got != "[1,2.5,-3]" {
		t.Errorf("vecToString = %q, want [1,2.5,-3]", got)
	}
	if got := vecToString(nil); got != "" {
		t.Errorf("vecToString(nil) = %q, want empty", got)
	}
}

func TestVecArg(t *testing.T) {
	if vecArg("") != nil {
		t.Error("vecArg(\"\") should be nil")
	}
	if vecArg("[1,2]") != "[1,2]" {
		t.Error("vecArg should pass through a non-empty literal")
	}
}

func TestNullableHelpers(t *testing.T) {
	if nullable("") != nil {
		t.Error("nullable(\"\") should be nil")
	}
	if nullable("x") != "x" {
		t.Error("nullable(\"x\") should be \"x\"")
	}
	if nullablePtr(nil) != nil {
		t.Error("nullablePtr(nil) should be nil")
	}
	empty := ""
	if nullablePtr(&empty) != nil {
		t.Error("nullablePtr(&\"\") should be nil")
	}
	v := "2024-01-15"
	if nullablePtr(&v) != "2024-01-15" {
		t.Error("nullablePtr should deref a non-empty pointer")
	}
}

func TestEmbed(t *testing.T) {
	ctx := context.Background()

	// nil embedder → empty literal, no error.
	s := &PgIngestStore{embedder: nil}
	if got, err := s.embed(ctx, "hello"); err != nil || got != "" {
		t.Errorf("nil embedder: got %q err %v, want empty/nil", got, err)
	}

	// blank text → empty literal even with an embedder.
	s = &PgIngestStore{embedder: stubEmbedder{vec: []float32{1, 2}}}
	if got, err := s.embed(ctx, "   "); err != nil || got != "" {
		t.Errorf("blank text: got %q err %v, want empty/nil", got, err)
	}

	// real text → pgvector literal.
	if got, err := s.embed(ctx, "hello"); err != nil || got != "[1,2]" {
		t.Errorf("embed: got %q err %v, want [1,2]", got, err)
	}

	// embedder error propagates.
	s = &PgIngestStore{embedder: stubEmbedder{err: context.Canceled}}
	if _, err := s.embed(ctx, "hello"); err == nil {
		t.Error("embed should propagate embedder error")
	}
}

// TestPgIngestStore_Integration round-trips a CreatePreference against real
// Postgres. Skipped unless MEMORYINGEST_TEST_DSN points at a DB with the
// user_memories schema and a seedable users row (FK). Uses a nil embedder so
// no OpenAI quota is needed; the vector columns are written NULL.
func TestPgIngestStore_Integration(t *testing.T) {
	dsn := os.Getenv("MEMORYINGEST_TEST_DSN")
	if dsn == "" {
		t.Skip("set MEMORYINGEST_TEST_DSN to run the ingest store integration test")
	}
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer pool.Close()

	const uid = "memoryingest-it"
	// Best-effort seed of the FK parent + cleanup. The test DB must allow it.
	_, _ = pool.Exec(ctx, `INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING`, uid)
	t.Cleanup(func() {
		_, _ = pool.Exec(ctx, `DELETE FROM user_memories WHERE user_id = $1`, uid)
		_, _ = pool.Exec(ctx, `DELETE FROM users WHERE id = $1`, uid)
	})

	store := NewPgIngestStore(pool, nil)
	id, err := store.CreatePreference(ctx, uid, CreatePreferenceInput{Suggestions: "prefers concise answers", Type: "communication"})
	if err != nil {
		t.Fatalf("CreatePreference: %v", err)
	}
	if id == "" {
		t.Fatal("CreatePreference returned empty id")
	}

	var (
		directives string
		layer      string
	)
	if err := pool.QueryRow(ctx, `
		SELECT p.conclusion_directives, m.memory_layer
		FROM user_memories_preferences p
		JOIN user_memories m ON m.id = p.user_memory_id
		WHERE p.id = $1 AND p.user_id = $2`, id, uid).Scan(&directives, &layer); err != nil {
		t.Fatalf("read back: %v", err)
	}
	if !strings.Contains(directives, "concise") {
		t.Errorf("conclusion_directives = %q", directives)
	}
	if layer != "preference" {
		t.Errorf("parent memory_layer = %q, want preference", layer)
	}
}

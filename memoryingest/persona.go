package memoryingest

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/json"
	"fmt"
	"log/slog"
	"math/big"
	"regexp"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/riverqueue/river"
)

// QueuePersonaRefresh is the River queue name for persona refresh
// jobs. The constant lives in egent-jobs/asynctask so all queue
// names are visible from a single place. Per-user flowControl
// parallelism = 1 (matches the QStash cron that the LobeHub
// implementation uses today).

// PersonaRefreshArgs is the River job payload. The BFF producer
// inserts one job per user per refresh cycle.
type PersonaRefreshArgs struct {
	UserID      string `json:"userId"`
	WorkspaceID string `json:"workspaceId,omitempty"`
	// Since is the floor for "new memories to merge into the persona".
	// Empty means "all memories since the last refresh".
	Since string `json:"since,omitempty"`
}

func (PersonaRefreshArgs) Kind() string { return "persona_refresh" }

// PersonaStore is the subset of palace.Store + extra persona-specific
// operations. Using a local interface avoids an import cycle between
// egent-jobs and egent-lobehub/memory/palace.
type PersonaStore interface {
	// ReadRecentMemories returns a flat list of recent memories for
	// the user, ready to be folded into a persona prompt.
	ReadRecentMemories(ctx context.Context, userID string, since time.Time, limit int) ([]MemoryRow, error)
	// ReadLatestPersona returns the current persona document for the
	// user (or nil if none exists yet).
	ReadLatestPersona(ctx context.Context, userID string) (*PersonaDoc, error)
	// WritePersona upserts the new persona document + appends a
	// history row. The implementation bumps version and computes
	// the diff against the previous version.
	WritePersona(ctx context.Context, userID string, persona, tagline string, memoryIDs []string) (*PersonaDoc, error)
}

// MemoryRow is a flat representation of one palace memory used as
// input to the persona refresh prompt.
type MemoryRow struct {
	ID       string `json:"id"`
	Layer    string `json:"layer"`
	Type     string `json:"type,omitempty"`
	Summary  string `json:"summary,omitempty"`
	Details  string `json:"details,omitempty"`
}

// PersonaDoc mirrors the LobeHub user_memory_persona_documents row.
type PersonaDoc struct {
	ID      string    `json:"id"`
	UserID  string    `json:"userId"`
	Profile string    `json:"profile"`
	Tagline string    `json:"tagline"`
	Persona string    `json:"persona"`
	Version int       `json:"version"`
	MemoryIDs []string `json:"memoryIds,omitempty"`
	CreatedAt time.Time `json:"createdAt"`
	UpdatedAt time.Time `json:"updatedAt"`
}

// ---------- worker ----------

// PersonaWorker refreshes a user's persona document by merging
// recently-extracted memories into the existing persona (or creating
// a new one when none exists).
type PersonaWorker struct {
	river.WorkerDefaults[PersonaRefreshArgs]

	pool   *pgxpool.Pool
	store  PersonaStore
	llm    LLMClient
	logger *slog.Logger
}

// Config configures the persona worker. The store may be nil in tests
// (the worker uses an inline fakeStore equivalent).
type PersonaConfig struct {
	Pool   *pgxpool.Pool
	Store  PersonaStore
	LLM    LLMClient
	Logger *slog.Logger
}

func NewPersonaWorker(cfg PersonaConfig) *PersonaWorker {
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	return &PersonaWorker{
		pool:   cfg.Pool,
		store:  cfg.Store,
		llm:    cfg.LLM,
		logger: cfg.Logger,
	}
}

// Work satisfies river.Worker.
func (w *PersonaWorker) Work(ctx context.Context, job *river.Job[PersonaRefreshArgs]) error {
	log := w.logger.With(
		"job_id", job.ID,
		"kind", job.Kind,
		"attempt", job.Attempt,
		"user_id", job.Args.UserID,
	)
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()

	if err := w.run(ctx, job.Args, log); err != nil {
		log.Error("persona refresh failed", "error", err)
		return err
	}
	log.Info("persona refresh succeeded")
	return nil
}

func (w *PersonaWorker) run(ctx context.Context, args PersonaRefreshArgs, log *slog.Logger) error {
	if w.store == nil {
		return fmt.Errorf("persona: store not configured")
	}
	if w.llm == nil {
		return fmt.Errorf("persona: LLM client not configured")
	}

	var since time.Time
	if args.Since != "" {
		t, err := time.Parse(time.RFC3339, args.Since)
		if err != nil {
			return fmt.Errorf("persona: parse since: %w", err)
		}
		since = t
	}

	memories, err := w.store.ReadRecentMemories(ctx, args.UserID, since, 100)
	if err != nil {
		return fmt.Errorf("persona: read memories: %w", err)
	}
	if len(memories) == 0 {
		log.Info("no new memories, skipping persona refresh")
		return nil
	}

	current, err := w.store.ReadLatestPersona(ctx, args.UserID)
	if err != nil {
		return fmt.Errorf("persona: read current: %w", err)
	}

	systemPrompt, userPrompt := buildPersonaPrompts(current, memories)

	reply, err := w.llm.Chat(ctx, systemPrompt, userPrompt)
	if err != nil {
		return fmt.Errorf("persona: LLM chat: %w", err)
	}

	tagline, persona := parsePersonaReply(reply, current)
	if strings.TrimSpace(persona) == "" {
		log.Warn("persona: LLM returned empty persona, skipping write")
		return nil
	}

	memoryIDs := make([]string, 0, len(memories))
	for _, m := range memories {
		memoryIDs = append(memoryIDs, m.ID)
	}
	if _, err := w.store.WritePersona(ctx, args.UserID, persona, tagline, memoryIDs); err != nil {
		return fmt.Errorf("persona: write: %w", err)
	}
	log.Info("persona refreshed",
		"memory_count", len(memories),
		"persona_length", len(persona))
	return nil
}

// ---------- prompts ----------

const personaSystemPrompt = `You are a persona synthesis assistant.

Given the user's existing persona document (may be empty) and a batch
of recently-extracted memories (identity / activity / context /
experience / preference), produce a fresh, compact persona that
incorporates the new information while preserving any details from
the previous persona that are still relevant.

Reply ONLY with a JSON object of shape:
{
  "tagline": "<one short sentence that captures the user's identity in a glance>",
  "persona": "<a few paragraphs of structured prose describing the user — keep under 800 words>"
}

If the previous persona is empty, build from scratch using only the
new memories. If both inputs are empty, return {"tagline":"","persona":""}.`

func buildPersonaPrompts(current *PersonaDoc, memories []MemoryRow) (string, string) {
	var prevSection strings.Builder
	if current != nil {
		fmt.Fprintf(&prevSection, "Previous persona (v%d, captured %s):\n",
			current.Version, current.UpdatedAt.Format(time.RFC3339))
		if current.Tagline != "" {
			fmt.Fprintf(&prevSection, "Tagline: %s\n", current.Tagline)
		}
		prevSection.WriteString("Persona:\n")
		prevSection.WriteString(current.Persona)
	} else {
		prevSection.WriteString("Previous persona: (none yet)")
	}

	var memSection strings.Builder
	fmt.Fprintf(&memSection, "\n\nNew memories (%d):\n", len(memories))
	for i, m := range memories {
		fmt.Fprintf(&memSection, "- [%s/%s] %s", m.Layer, m.Type, m.Summary)
		if m.Details != "" {
			fmt.Fprintf(&memSection, " — %s", m.Details)
		}
		memSection.WriteString("\n")
		if i > 50 {
			memSection.WriteString("… (truncated)\n")
			break
		}
	}

	return personaSystemPrompt, prevSection.String() + memSection.String()
}

func parsePersonaReply(reply string, current *PersonaDoc) (tagline, persona string) {
	match := personaJSON.FindString(reply)
	if match == "" {
		// Fallback: treat the entire reply as the persona body.
		if current != nil {
			return current.Tagline, reply
		}
		return "", reply
	}
	var out struct {
		Tagline string `json:"tagline"`
		Persona string `json:"persona"`
	}
	if err := json.Unmarshal([]byte(match), &out); err != nil {
		if current != nil {
			return current.Tagline, reply
		}
		return "", reply
	}
	return out.Tagline, out.Persona
}

var personaJSON = regexp.MustCompile(`\{[^{}]*"(tagline|persona)"[^{}]*\}`)

// ---------- Postgres implementation ----------

// PgPersonaStore is the Postgres implementation of PersonaStore.
// Reads from user_memories + user_memories_*; writes to
// user_memory_persona_documents and user_memory_persona_document_histories.
type PgPersonaStore struct {
	pool *pgxpool.Pool
}

func NewPgPersonaStore(pool *pgxpool.Pool) *PgPersonaStore {
	return &PgPersonaStore{pool: pool}
}

// ReadRecentMemories satisfies PersonaStore.
func (s *PgPersonaStore) ReadRecentMemories(ctx context.Context, userID string, since time.Time, limit int) ([]MemoryRow, error) {
	if limit <= 0 {
		limit = 100
	}
	rows, err := s.pool.Query(ctx, `
		SELECT id, memory_layer, COALESCE(memory_type, ''), COALESCE(summary, ''), COALESCE(details, '')
		FROM user_memories
		WHERE user_id = $1
		  AND status = 'active'
		  AND ($2::timestamptz IS NULL OR captured_at >= $2)
		ORDER BY captured_at DESC
		LIMIT $3`, userID, nullableTime(since), limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []MemoryRow
	for rows.Next() {
		var m MemoryRow
		if err := rows.Scan(&m.ID, &m.Layer, &m.Type, &m.Summary, &m.Details); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// ReadLatestPersona satisfies PersonaStore.
func (s *PgPersonaStore) ReadLatestPersona(ctx context.Context, userID string) (*PersonaDoc, error) {
	var p PersonaDoc
	var memoryIDsJSON []byte
	err := s.pool.QueryRow(ctx, `
		SELECT id, user_id, profile, COALESCE(tagline, ''), COALESCE(persona, ''),
		       version, COALESCE(memory_ids, 'null'::jsonb), created_at, updated_at
		FROM user_memory_persona_documents
		WHERE user_id = $1 AND profile = 'default'
		ORDER BY version DESC
		LIMIT 1`, userID).Scan(
		&p.ID, &p.UserID, &p.Profile, &p.Tagline, &p.Persona,
		&p.Version, &memoryIDsJSON, &p.CreatedAt, &p.UpdatedAt)
	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, nil
		}
		return nil, err
	}
	if len(memoryIDsJSON) > 0 {
		_ = json.Unmarshal(memoryIDsJSON, &p.MemoryIDs)
	}
	return &p, nil
}

// WritePersona satisfies PersonaStore. Bumps the version and writes
// a history row in the same transaction.
func (s *PgPersonaStore) WritePersona(ctx context.Context, userID, persona, tagline string, memoryIDs []string) (*PersonaDoc, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tx.Rollback(ctx) }()

	// Look up the current version (if any) for the diff.
	var prev PersonaDoc
	err = tx.QueryRow(ctx, `
		SELECT id, version, COALESCE(tagline, ''), COALESCE(persona, '')
		FROM user_memory_persona_documents
		WHERE user_id = $1 AND profile = 'default'
		ORDER BY version DESC
		LIMIT 1`, userID).Scan(&prev.ID, &prev.Version, &prev.Tagline, &prev.Persona)
	if err != nil && err != pgx.ErrNoRows {
		return nil, err
	}

	nextVersion := 1
	if err != pgx.ErrNoRows {
		nextVersion = prev.Version + 1
	}

	personaID, err := randomID()
	if err != nil {
		return nil, err
	}
	historyID, err := randomID()
	if err != nil {
		return nil, err
	}
	memoryIDsJSON, err := json.Marshal(memoryIDs)
	if err != nil {
		return nil, err
	}

	if _, err := tx.Exec(ctx, `
		INSERT INTO user_memory_persona_documents
			(id, user_id, profile, tagline, persona, memory_ids, version)
		VALUES ($1, $2, 'default', $3, $4, $5::jsonb, $6)
		ON CONFLICT (user_id, profile) DO UPDATE
		SET tagline = EXCLUDED.tagline,
		    persona = EXCLUDED.persona,
		    memory_ids = EXCLUDED.memory_ids,
		    version = EXCLUDED.version,
		    updated_at = now()`,
		personaID, userID, tagline, persona, string(memoryIDsJSON), nextVersion,
	); err != nil {
		return nil, err
	}

	// Append history row.
	if _, err := tx.Exec(ctx, `
		INSERT INTO user_memory_persona_document_histories
			(id, persona_id, user_id, profile, snapshot_persona, snapshot_tagline,
			 reasoning, diff_persona, diff_tagline, edited_by, memory_ids, source_ids,
			 previous_version, next_version)
		VALUES ($1, $2, $3, 'default', $4, $5, $6, $7, $8, 'agent',
		        $9::jsonb, '[]'::jsonb, $10, $11)`,
		historyID,
		personaID, userID,
		persona, tagline,
		"agent refresh", diffOrEmpty(prev.Persona, persona), diffOrEmpty(prev.Tagline, tagline),
		string(memoryIDsJSON),
		prev.Version, nextVersion,
	); err != nil {
		return nil, err
	}

	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return &PersonaDoc{
		ID: personaID, UserID: userID, Profile: "default",
		Tagline: tagline, Persona: persona, Version: nextVersion,
		MemoryIDs: memoryIDs,
		CreatedAt: time.Now(), UpdatedAt: time.Now(),
	}, nil
}

func nullableTime(t time.Time) any {
	if t.IsZero() {
		return nil
	}
	return t
}

func diffOrEmpty(old, new string) string {
	if old == new {
		return ""
	}
	return new
}

func randomID() (string, error) {
	const alphabet = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
	const size = 21
	b := make([]byte, size)
	for i := range b {
		n, err := rand.Int(rand.Reader, big.NewInt(int64(len(alphabet))))
		if err != nil {
			return "", err
		}
		b[i] = alphabet[n.Int64()]
	}
	return string(b), nil
}

// ---------- compile-time checks ----------

var _ PersonaStore = (*PgPersonaStore)(nil)

// Ensure the sql import is referenced even if some downstream consumers
// don't use database/sql directly. This guards against an unused-import
// error if the Postgres impl is later replaced.
var _ sql.IsolationLevel = sql.LevelDefault
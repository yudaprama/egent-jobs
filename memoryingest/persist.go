package memoryingest

import (
	"context"
	"fmt"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"egent-jobs/embeddings"
)

// PgIngestStore is the Postgres implementation of [Store]. It writes the
// parent user_memories row plus one child user_memories_* row in a single
// transaction (so a partial failure can't orphan a parent), and embeds the
// layer's primary text into the parent summary_vector_1024 + the layer
// vector column via the configured embedder.
//
// It deliberately writes directly to the pool — matching the egent-jobs
// convention (asynctask.Store, PgPersonaStore) — rather than importing
// egent-lobehub/memory/palace, so the two submodules stay decoupled. The
// DB schema (not Go code) is the shared contract; the INSERTs mirror
// palace.PgStore exactly.
type PgIngestStore struct {
	pool     *pgxpool.Pool
	embedder embeddings.Embedder
}

// NewPgIngestStore builds an ingest store. A nil embedder is allowed —
// the *_vector columns are then written NULL (search degrades to the
// ILIKE/recency path in palace).
func NewPgIngestStore(pool *pgxpool.Pool, embedder embeddings.Embedder) *PgIngestStore {
	return &PgIngestStore{pool: pool, embedder: embedder}
}

// embed returns the pgvector literal for text, or "" when no embedder is
// configured or text is blank.
func (s *PgIngestStore) embed(ctx context.Context, text string) (string, error) {
	if s.embedder == nil || strings.TrimSpace(text) == "" {
		return "", nil
	}
	res, err := s.embedder.EmbedBatch(ctx, []embeddings.Input{{ChunkID: "m", Text: text}})
	if err != nil {
		return "", fmt.Errorf("memoryingest: embed: %w", err)
	}
	if len(res) == 0 || len(res[0].Vector) == 0 {
		return "", nil
	}
	return vecToString(res[0].Vector), nil
}

// insertParent writes the user_memories row and returns its generated id.
func (s *PgIngestStore) insertParent(ctx context.Context, tx pgx.Tx, userID, layer, summary, vec string) (string, error) {
	id, err := randomID()
	if err != nil {
		return "", fmt.Errorf("memoryingest: gen parent id: %w", err)
	}
	const q = `
		INSERT INTO user_memories
			(id, user_id, memory_layer, status, summary, summary_vector_1024)
		VALUES ($1, $2, $3, 'active', $4, $5::vector)`
	if _, err := tx.Exec(ctx, q, id, userID, layer, nullable(summary), vecArg(vec)); err != nil {
		return "", fmt.Errorf("memoryingest: insert user_memories: %w", err)
	}
	return id, nil
}

// runInsert wraps the common begin/commit + parent-insert flow shared by
// every layer. childInsert receives the open tx and the parent id and must
// insert exactly one child row, returning its id.
func (s *PgIngestStore) runInsert(ctx context.Context, userID, layer, summary, vec string, childInsert func(tx pgx.Tx, parentID string) (string, error)) (string, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return "", fmt.Errorf("memoryingest: begin: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	parentID, err := s.insertParent(ctx, tx, userID, layer, summary, vec)
	if err != nil {
		return "", err
	}
	childID, err := childInsert(tx, parentID)
	if err != nil {
		return "", err
	}
	if err := tx.Commit(ctx); err != nil {
		return "", fmt.Errorf("memoryingest: commit: %w", err)
	}
	return childID, nil
}

// CreateIdentity satisfies [Store].
func (s *PgIngestStore) CreateIdentity(ctx context.Context, userID string, in CreateIdentityInput) (string, error) {
	vec, err := s.embed(ctx, in.Description)
	if err != nil {
		return "", err
	}
	return s.runInsert(ctx, userID, "identity", in.Description, vec, func(tx pgx.Tx, parentID string) (string, error) {
		id, err := randomID()
		if err != nil {
			return "", err
		}
		const q = `
			INSERT INTO user_memories_identities
				(id, user_id, user_memory_id, description, role, relationship, type, episodic_date, tags, description_vector)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector)`
		if _, err := tx.Exec(ctx, q,
			id, userID, parentID,
			nullable(in.Description), nullable(in.Role), nullable(in.Relationship),
			nullable(in.Type), nullablePtr(in.EpisodicDate), in.Tags, vecArg(vec),
		); err != nil {
			return "", fmt.Errorf("memoryingest: insert identity: %w", err)
		}
		return id, nil
	})
}

// CreateActivity satisfies [Store].
func (s *PgIngestStore) CreateActivity(ctx context.Context, userID string, in CreateActivityInput) (string, error) {
	vec, err := s.embed(ctx, in.Narrative)
	if err != nil {
		return "", err
	}
	status := in.Status
	if status == "" {
		status = "pending"
	}
	return s.runInsert(ctx, userID, "activity", in.Narrative, vec, func(tx pgx.Tx, parentID string) (string, error) {
		id, err := randomID()
		if err != nil {
			return "", err
		}
		const q = `
			INSERT INTO user_memories_activities
				(id, user_id, user_memory_id, type, status, notes, narrative, tags, narrative_vector)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector)`
		if _, err := tx.Exec(ctx, q,
			id, userID, parentID,
			in.Type, status, nullable(in.Notes), nullable(in.Narrative), in.Tags, vecArg(vec),
		); err != nil {
			return "", fmt.Errorf("memoryingest: insert activity: %w", err)
		}
		return id, nil
	})
}

// CreateContext satisfies [Store]. Contexts reference parent memories via
// the JSONB user_memory_ids array (no FK column), so the parent id is
// seeded as the first entry.
func (s *PgIngestStore) CreateContext(ctx context.Context, userID string, in CreateContextInput) (string, error) {
	vec, err := s.embed(ctx, in.Title)
	if err != nil {
		return "", err
	}
	return s.runInsert(ctx, userID, "context", in.Title, vec, func(tx pgx.Tx, parentID string) (string, error) {
		id, err := randomID()
		if err != nil {
			return "", err
		}
		const q = `
			INSERT INTO user_memories_contexts
				(id, user_id, user_memory_ids, title, description, type, tags, description_vector)
			VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7, $8::vector)`
		idsJSON := fmt.Sprintf(`["%s"]`, parentID)
		if _, err := tx.Exec(ctx, q,
			id, userID, idsJSON,
			in.Title, nullable(in.Description), nullable(in.Type), in.Tags, vecArg(vec),
		); err != nil {
			return "", fmt.Errorf("memoryingest: insert context: %w", err)
		}
		return id, nil
	})
}

// CreateExperience satisfies [Store].
func (s *PgIngestStore) CreateExperience(ctx context.Context, userID string, in CreateExperienceInput) (string, error) {
	vec, err := s.embed(ctx, in.Situation)
	if err != nil {
		return "", err
	}
	return s.runInsert(ctx, userID, "experience", in.Situation, vec, func(tx pgx.Tx, parentID string) (string, error) {
		id, err := randomID()
		if err != nil {
			return "", err
		}
		const q = `
			INSERT INTO user_memories_experiences
				(id, user_id, user_memory_id, type, situation, action, key_learning, tags, situation_vector)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector)`
		if _, err := tx.Exec(ctx, q,
			id, userID, parentID,
			nullable(in.Type), nullable(in.Situation), nullable(in.Action), nullable(in.KeyLearning), in.Tags, vecArg(vec),
		); err != nil {
			return "", fmt.Errorf("memoryingest: insert experience: %w", err)
		}
		return id, nil
	})
}

// CreatePreference satisfies [Store]. The extractor only yields the
// directive text ("suggestions" in its JSON), which palace treats as the
// preference's primary conclusion_directives field.
func (s *PgIngestStore) CreatePreference(ctx context.Context, userID string, in CreatePreferenceInput) (string, error) {
	vec, err := s.embed(ctx, in.Suggestions)
	if err != nil {
		return "", err
	}
	return s.runInsert(ctx, userID, "preference", in.Suggestions, vec, func(tx pgx.Tx, parentID string) (string, error) {
		id, err := randomID()
		if err != nil {
			return "", err
		}
		const q = `
			INSERT INTO user_memories_preferences
				(id, user_id, user_memory_id, conclusion_directives, type, tags, conclusion_directives_vector)
			VALUES ($1, $2, $3, $4, $5, $6, $7::vector)`
		if _, err := tx.Exec(ctx, q,
			id, userID, parentID,
			nullable(in.Suggestions), nullable(in.Type), in.Tags, vecArg(vec),
		); err != nil {
			return "", fmt.Errorf("memoryingest: insert preference: %w", err)
		}
		return id, nil
	})
}

// vecToString formats a []float32 as a pgvector literal `[a,b,c]`.
func vecToString(v []float32) string {
	if len(v) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteByte('[')
	for i, f := range v {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%g", f)
	}
	b.WriteByte(']')
	return b.String()
}

// vecArg returns the vector literal for binding, or nil so the column
// stores SQL NULL when no embedding was produced.
func vecArg(vec string) any {
	if vec == "" {
		return nil
	}
	return vec
}

// nullable returns nil for an empty string so the column stores NULL.
func nullable(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// nullablePtr returns nil for a nil/empty *string, else the value.
func nullablePtr(s *string) any {
	if s == nil || *s == "" {
		return nil
	}
	return *s
}

// compile-time check that PgIngestStore satisfies Store.
var _ Store = (*PgIngestStore)(nil)

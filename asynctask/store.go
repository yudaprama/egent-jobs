package asynctask

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Store is the thin DB-access layer the workers use to mirror their progress
// into the existing async_tasks table. Each method corresponds to one of the
// `AsyncTaskModel.update` calls in lobehub/apps/server/src/routers/async/file.ts.
//
// The pool is shared with the River client. Callers pass the async_task ID
// (a uuid) — River's own job row lives in river_job, async_tasks stays as the
// user-visible status ledger that the BFF still reads via Tier 1 pREST CRUD.
type Store struct {
	pool *pgxpool.Pool
}

// NewStore returns a Store bound to the given pool.
func NewStore(pool *pgxpool.Pool) *Store {
	return &Store{pool: pool}
}

// MarkProcessing flips async_tasks.status to "processing". Mirrors the first
// `asyncTaskModel.update(taskId, { status: AsyncTaskStatus.Processing })` call
// in async/file.ts:108.
func (s *Store) MarkProcessing(ctx context.Context, taskID string) error {
	return s.updateStatus(ctx, taskID, StatusProcessing, nil)
}

// MarkSuccess flips async_tasks.status to "success" and records the run
// duration in milliseconds. Mirrors async/file.ts:161-164.
func (s *Store) MarkSuccess(ctx context.Context, taskID string, durationMS int64) error {
	_, err := s.pool.Exec(ctx, `
		UPDATE public.async_tasks
		   SET status = $2, duration = $3, updated_at = NOW()
		 WHERE id = $1`,
		taskID, StatusSuccess, durationMS)
	if err != nil {
		return fmt.Errorf("asynctask: mark success %s: %w", taskID, err)
	}
	return nil
}

// MarkError flips async_tasks.status to "error" and persists the structured
// error payload. Mirrors async/file.ts:174-177.
func (s *Store) MarkError(ctx context.Context, taskID string, errType, message string) error {
	payload := &Error{Name: firstNonEmpty(errType, ErrorTypeGeneric), Message: message}
	return s.updateStatus(ctx, taskID, StatusError, payload)
}

func (s *Store) updateStatus(ctx context.Context, taskID, status string, errPayload *Error) error {
	var errJSON []byte
	if errPayload != nil {
		b, err := json.Marshal(errPayload)
		if err != nil {
			return fmt.Errorf("asynctask: marshal error payload: %w", err)
		}
		errJSON = b
	}
	tag, err := s.pool.Exec(ctx, `
		UPDATE public.async_tasks
		   SET status = $2, error = COALESCE($3, error), updated_at = NOW()
		 WHERE id = $1`,
		taskID, status, errJSON)
	if err != nil {
		return fmt.Errorf("asynctask: update %s to %s: %w", taskID, status, err)
	}
	if tag.RowsAffected() == 0 {
		return fmt.Errorf("asynctask: %w for id %s", pgx.ErrNoRows, taskID)
	}
	return nil
}

// ResolveWorkspaceIDFromFile mirrors resolveWorkspaceIdFromFile in
// async/file.ts:44-58 — it fetches the file's workspace_id (used for tenancy
// scoping on subsequent reads). Returns empty string if the file has no
// workspace (personal scope).
func (s *Store) ResolveWorkspaceIDFromFile(ctx context.Context, userID, fileID string) (string, error) {
	if userID == "" || fileID == "" {
		return "", nil
	}
	var ownerID, workspaceID *string
	err := s.pool.QueryRow(ctx,
		`SELECT user_id, workspace_id FROM public.files WHERE id = $1`, fileID,
	).Scan(&ownerID, &workspaceID)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("asynctask: resolve workspace for %s: %w", fileID, err)
	}
	if ownerID == nil || *ownerID != userID {
		// Caller does not own the file — do not leak workspace_id.
		return "", nil
	}
	if workspaceID == nil {
		return "", nil
	}
	return *workspaceID, nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

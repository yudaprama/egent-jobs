// Package fileingest implements the River worker that replaces the
// embeddingChunks procedure in lobehub/apps/server/src/routers/async/file.ts.
//
// On the legacy path the BFF enqueues an async_tasks row, then the Next.js
// route handler does pMap'd batched HTTP calls to the embedding model and
// writes to public.embeddings. This worker subsumes that loop: it owns the
// status transitions on async_tasks, batches the embedding call, and bulk
// inserts the resulting vectors. The BFF becomes a thin producer that
// inserts a River job (and a single async_tasks row) instead of calling
// itself over HTTP.
package fileingest

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"strconv"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/riverqueue/river"

	"egent-jobs/asynctask"
	"egent-jobs/embeddings"
)

// EmbedFileChunksArgs is the River JobArgs payload. It captures the same
// fields the BFF currently passes into embeddingChunks:
//
//	taskId     — async_tasks row whose status we are driving
//	fileId     — the file whose chunks need vectors
//	userId     — owner (read for tenant scoping on the chunks/embeddings reads)
//	workspaceId — optional workspace scope (mirrors async/file.ts:71-77)
type EmbedFileChunksArgs struct {
	TaskID      string `json:"taskId"`
	FileID      string `json:"fileId"`
	UserID      string `json:"userId"`
	WorkspaceID string `json:"workspaceId,omitempty"`
}

// Kind is the registered River job kind identifier. Exported because the
// producer (BFF or any other service that enqueues the job) needs to pass
// the same string to client.Insert via river.NewJobArgsWithKind.
func (EmbedFileChunksArgs) Kind() string { return "embed_file_chunks" }

// EmbedFileChunksWorker embeds the chunks produced for a file and writes them
// to public.embeddings. The flow is:
//
//  1. Mark async_tasks.status = "processing" (matches async/file.ts:108).
//  2. Resolve workspace_id if the caller didn't pass it (async/file.ts:44-58).
//  3. Fetch all chunks for the file, batched by batchSize (EMBEDDING_BATCH_SIZE
//     in the TS code).
//  4. For each batch, call the embedder with concurrency=concurrency
//     (EMBEDDING_CONCURRENCY in the TS code) and bulk-insert the resulting
//     vectors.
//  5. Mark async_tasks.status = "success" with the elapsed duration
//     (async/file.ts:161-164). On error, mark "error" with the structured
//     AsyncTaskError payload (async/file.ts:174-177).
type EmbedFileChunksWorker struct {
	river.WorkerDefaults[EmbedFileChunksArgs]

	pool    *pgxpool.Pool
	store   *asynctask.Store
	embed   embeddings.Embedder
	logger  *slog.Logger
	batch   int
	conc    int
	timeout time.Duration
}

// WorkerConfig configures the worker. Defaults are chosen to match the
// values used by the legacy TS code (fileEnv.EMBEDDING_BATCH_SIZE,
// fileEnv.EMBEDDING_CONCURRENCY, ASYNC_TASK_TIMEOUT).
type WorkerConfig struct {
	Pool         *pgxpool.Pool
	Store        *asynctask.Store
	Embedder     embeddings.Embedder
	Logger       *slog.Logger
	BatchSize    int           // default 10
	Concurrency  int           // default 3
	JobTimeout   time.Duration // default 15m
}

// NewEmbedFileChunksWorker constructs a worker. The logger is wrapped in
// fromContext() so the default River job-row fields (id, attempt) are
// attached on every log line.
func NewEmbedFileChunksWorker(cfg WorkerConfig) *EmbedFileChunksWorker {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 10
	}
	if cfg.Concurrency <= 0 {
		cfg.Concurrency = 3
	}
	if cfg.JobTimeout <= 0 {
		cfg.JobTimeout = 15 * time.Minute
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	return &EmbedFileChunksWorker{
		pool:    cfg.Pool,
		store:   cfg.Store,
		embed:   cfg.Embedder,
		logger:  cfg.Logger,
		batch:   cfg.BatchSize,
		conc:    cfg.Concurrency,
		timeout: cfg.JobTimeout,
	}
}

// Work satisfies river.Worker.
func (w *EmbedFileChunksWorker) Work(ctx context.Context, job *river.Job[EmbedFileChunksArgs]) error {
	log := w.logger.With(
		"job_id", job.ID,
		"kind", job.Kind,
		"attempt", job.Attempt,
		"task_id", job.Args.TaskID,
		"file_id", job.Args.FileID,
		"user_id", job.Args.UserID,
	)

	// ASYNC_TASK_TIMEOUT: the BFF rejects the work after 15 minutes. We
	// mirror that with a context timeout so River also stops scheduling
	// the job at the same bound.
	ctx, cancel := context.WithTimeout(ctx, w.timeout)
	defer cancel()

	if err := w.store.MarkProcessing(ctx, job.Args.TaskID); err != nil {
		// async_tasks row vanished — likely a delete cascade. Cancel so we
		// don't retry forever on a row that no longer exists.
		if errors.Is(err, pgx.ErrNoRows) {
			log.Warn("async_tasks row missing; cancelling")
			return river.JobCancel(fmt.Errorf("asynctask %s not found: %w", job.Args.TaskID, err))
		}
		log.Error("mark processing failed", "error", err)
		return err
	}

	workspaceID := job.Args.WorkspaceID
	if workspaceID == "" {
		// Lazy resolve: matches resolveWorkspaceIdFromFile in async/file.ts.
		// We only trust the result if the file is owned by job.Args.UserID
		// (enforced inside ResolveWorkspaceIDFromFile).
		resolved, err := w.store.ResolveWorkspaceIDFromFile(ctx, job.Args.UserID, job.Args.FileID)
		if err != nil {
			log.Warn("resolve workspace failed; continuing with personal scope", "error", err)
		} else {
			workspaceID = resolved
		}
	}

	start := time.Now()
	if err := w.run(ctx, job.Args, workspaceID, log); err != nil {
		log.Error("embed failed", "error", err, "duration_ms", time.Since(start).Milliseconds())
		// Match the TS error shape: AsyncTaskError(name, message) → JSON.
		// The TS handler names the failure after the upstream error's name
		// and falls back to AsyncTaskError when missing. We use the constant
		// EmbeddingError directly because the BFF only ever calls into
		// embeddingChunks from this code path.
		if markErr := w.store.MarkError(ctx, job.Args.TaskID, asynctask.ErrorTypeEmbeddingError, err.Error()); markErr != nil {
			log.Error("mark error failed", "error", markErr)
		}
		return err
	}

	if err := w.store.MarkSuccess(ctx, job.Args.TaskID, time.Since(start).Milliseconds()); err != nil {
		log.Error("mark success failed", "error", err)
		return err
	}
	log.Info("embed succeeded", "duration_ms", time.Since(start).Milliseconds(), "workspace_id", workspaceID)
	return nil
}

// run is split out from Work so the SQL + embedding flow can be unit tested
// against a real Postgres without spinning up a River client.
func (w *EmbedFileChunksWorker) run(ctx context.Context, args EmbedFileChunksArgs, workspaceID string, log *slog.Logger) error {
	chunks, err := w.fetchChunks(ctx, args.UserID, args.FileID, workspaceID)
	if err != nil {
		return fmt.Errorf("fetch chunks: %w", err)
	}
	if len(chunks) == 0 {
		log.Warn("no chunks found; nothing to embed")
		return nil
	}
	log.Info("chunks fetched", "count", len(chunks))

	batches := chunkBy(chunks, w.batch)
	log.Info("embedding batches", "batches", len(batches), "concurrency", w.conc, "batch_size", w.batch)

	// pMap-equivalent: a worker pool of size w.conc processes the batches.
	// Errors from any batch cancel the rest of the work via context.
	jobErrCh := make(chan error, len(batches))
	workCtx, cancelWork := context.WithCancel(ctx)
	defer cancelWork()
	slots := make(chan struct{}, w.conc)

	for idx, batch := range batches {
		idx, batch := idx, batch
		select {
		case <-workCtx.Done():
			return workCtx.Err()
		case slots <- struct{}{}:
		}
		go func() {
			defer func() { <-slots }()
			if err := w.embedAndPersist(workCtx, args, batch, log); err != nil {
				cancelWork()
				jobErrCh <- fmt.Errorf("batch %d: %w", idx, err)
			}
		}()
	}

	// Drain.
	for range batches {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case err := <-jobErrCh:
			if err != nil {
				return err
			}
		}
	}
	return nil
}

type chunkRow struct {
	ID   string
	Text string
}

func (w *EmbedFileChunksWorker) fetchChunks(ctx context.Context, userID, fileID, workspaceID string) ([]chunkRow, error) {
	// Mirrors chunkModel.getChunksTextByFileId in async/file.ts:117. We add
	// a tenancy check via user_id and the optional workspace_id so a worker
	// can never embed another tenant's chunks. file_id is sourced from
	// file_chunks (the chunking step has already joined chunks to the file
	// in that table).
	rows, err := w.pool.Query(ctx, `
		SELECT c.id::text, c.text
		  FROM public.chunks AS c
		  JOIN public.file_chunks AS fc ON fc.chunk_id = c.id
		 WHERE fc.file_id = $1
		   AND c.user_id = $2
		   AND ($3 = '' OR c.workspace_id::text = $3)
		 ORDER BY c."index" ASC`,
		fileID, userID, workspaceID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []chunkRow
	for rows.Next() {
		var c chunkRow
		if err := rows.Scan(&c.ID, &c.Text); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

func (w *EmbedFileChunksWorker) embedAndPersist(ctx context.Context, args EmbedFileChunksArgs, batch []chunkRow, log *slog.Logger) error {
	inputs := make([]embeddings.Input, len(batch))
	for i, c := range batch {
		inputs[i] = embeddings.Input{ChunkID: c.ID, Text: c.Text}
	}
	results, err := w.embed.EmbedBatch(ctx, inputs)
	if err != nil {
		return fmt.Errorf("embed: %w", err)
	}
	if len(results) != len(batch) {
		return fmt.Errorf("embed: expected %d results, got %d", len(batch), len(results))
	}
	if err := w.persistEmbeddings(ctx, args, results, log); err != nil {
		return fmt.Errorf("persist: %w", err)
	}
	return nil
}

func (w *EmbedFileChunksWorker) persistEmbeddings(ctx context.Context, args EmbedFileChunksArgs, results []embeddings.Result, log *slog.Logger) error {
	// public.embeddings has a unique constraint on chunk_id, so we use an
	// ON CONFLICT DO UPDATE to make the worker idempotent against retries.
	// This matches the BFF's ChunkService / EmbeddingModel.bulkCreate which
	// silently overwrites prior rows.
	batch := &pgx.Batch{}
	for _, r := range results {
		if len(r.Vector) != 1024 {
			return fmt.Errorf("vector dimension mismatch for chunk %s: got %d want 1024", r.ChunkID, len(r.Vector))
		}
		// pgvector's text format is "[v1,v2,...]". We hand-build the literal
		// rather than using the pgvector-go type to avoid an extra dep here.
		vecLit := vectorToPG(r.Vector)
		batch.Queue(`
			INSERT INTO public.embeddings
				(chunk_id, embeddings, model, user_id, workspace_id, client_id)
			VALUES
				($1, $2::vector, $3, $4, NULLIF($5, ''), NULL)
			ON CONFLICT (chunk_id) DO UPDATE
			SET embeddings = EXCLUDED.embeddings,
			    model = EXCLUDED.model,
			    updated_at = NOW()`,
			r.ChunkID, vecLit, w.embed.Model(), args.UserID, args.WorkspaceID)
	}
	br := w.pool.SendBatch(ctx, batch)
	defer br.Close()
	for range results {
		if _, err := br.Exec(); err != nil {
			return err
		}
	}
	log.Debug("embeddings persisted", "count", len(results), "model", w.embed.Model())
	return nil
}

func chunkBy[T any](items []T, size int) [][]T {
	if size <= 0 {
		size = 1
	}
	n := int(math.Ceil(float64(len(items)) / float64(size)))
	out := make([][]T, 0, n)
	for i := 0; i < len(items); i += size {
		end := i + size
		if end > len(items) {
			end = len(items)
		}
		out = append(out, items[i:end])
	}
	return out
}

func vectorToPG(v []float32) string {
	// pgvector's text format: "[1.0,2.0,3.0]" — no spaces, no leading
	// scientific notation quirks. We use a fixed-precision representation
	// because the column stores it as numeric anyway.
	b := make([]byte, 0, 1+len(v)*8)
	b = append(b, '[')
	for i, x := range v {
		if i > 0 {
			b = append(b, ',')
		}
		b = strconv.AppendFloat(b, float64(x), 'f', -1, 32)
	}
	b = append(b, ']')
	return string(b)
}

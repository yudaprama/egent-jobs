package fileingest

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/riverqueue/river"

	"egent-jobs/asynctask"
)

// ParseFileToChunksArgs is the River JobArgs payload that replaces the
// parseFileToChunks tRPC mutation in async/file.ts. On the legacy path the
// BFF downloaded the file, called ContentChunk.chunkContent (which shells
// out to unstructured.io), bulk-inserted chunks into public.chunks, and
// created a documents row. This worker owns the full lifecycle:
//
//  1. Mark async_tasks.status = "processing"
//  2. Fetch file bytes from the file service URL (AList)
//  3. Run the Chunker abstraction to produce chunk rows
//  4. Bulk INSERT into public.chunks + public.file_chunks
//  5. Mark async_tasks.status = "success"
//  6. Optionally enqueue an embed_file_chunks job (auto-embedding)
type ParseFileToChunksArgs struct {
	TaskID      string `json:"taskId"`
	FileID      string `json:"fileId"`
	UserID      string `json:"userId"`
	WorkspaceID string `json:"workspaceId,omitempty"`
	SkipExist   bool   `json:"skipExist,omitempty"`
}

func (ParseFileToChunksArgs) Kind() string { return "parse_file_to_chunks" }

// Chunker abstracts the document-chunking pipeline so the worker doesn't
// depend on unstructured.io or any specific JS library. A real implementation
// should replicate the behavior of lobehub's ContentChunk.chunkByDefault:
// download file → partition via unstructured → return structured chunks.
type Chunker interface {
	// Chunk splits raw file content into structured chunks. The returned
	// ChunkRows will be bulk-inserted into public.chunks. The caller owns
	// the async_tasks lifecycle; Chunker is purely a pure transform.
	Chunk(ctx context.Context, input ChunkInput) ([]ChunkRow, error)
}

type ChunkInput struct {
	Filename string
	FileType string
	Content  []byte
}

type ChunkRow struct {
	Text     string
	Metadata map[string]any
	Type     string
}

// Fetcher abstracts downloading file bytes from the BFF's file service URL.
// In the TS BFF this is fileService.getFileByteArray(file.url) which goes
// through Supabase Storage / AList. In Go we replicate it via a configurable
// HTTP fetch so the worker can be tested with a fake.
type Fetcher interface {
	Fetch(ctx context.Context, url string) ([]byte, error)
}

// ParseFileToChunksWorker ports async/file.ts parseFileToChunks to River.
type ParseFileToChunksWorker struct {
	river.WorkerDefaults[ParseFileToChunksArgs]

	pool    *pgxpool.Pool
	store   *asynctask.Store
	fetcher Fetcher
	chunker Chunker
	logger  *slog.Logger
	timeout time.Duration
}

type ParseWorkerConfig struct {
	Pool     *pgxpool.Pool
	Store    *asynctask.Store
	Fetcher  Fetcher
	Chunker  Chunker
	Logger   *slog.Logger
	Timeout  time.Duration
}

func NewParseFileToChunksWorker(cfg ParseWorkerConfig) *ParseFileToChunksWorker {
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 15 * time.Minute
	}
	if cfg.Fetcher == nil {
		cfg.Fetcher = &HTTPFetcher{}
	}
	if cfg.Chunker == nil {
		cfg.Chunker = &TextChunker{}
	}
	return &ParseFileToChunksWorker{
		pool:    cfg.Pool,
		store:   cfg.Store,
		fetcher: cfg.Fetcher,
		chunker: cfg.Chunker,
		logger:  cfg.Logger,
		timeout: cfg.Timeout,
	}
}

// Work satisfies river.Worker.
func (w *ParseFileToChunksWorker) Work(ctx context.Context, job *river.Job[ParseFileToChunksArgs]) error {
	log := w.logger.With(
		"job_id", job.ID,
		"kind", job.Kind,
		"attempt", job.Attempt,
		"task_id", job.Args.TaskID,
		"file_id", job.Args.FileID,
		"user_id", job.Args.UserID,
	)

	ctx, cancel := context.WithTimeout(ctx, w.timeout)
	defer cancel()

	if err := w.store.MarkProcessing(ctx, job.Args.TaskID); err != nil {
		if pgx.ErrNoRows == err {
			log.Warn("async_tasks row missing; cancelling")
			return river.JobCancel(fmt.Errorf("asynctask %s not found: %w", job.Args.TaskID, err))
		}
		log.Error("mark processing failed", "error", err)
		return err
	}

	workspaceID := job.Args.WorkspaceID
	if workspaceID == "" {
		resolved, err := w.store.ResolveWorkspaceIDFromFile(ctx, job.Args.UserID, job.Args.FileID)
		if err == nil {
			workspaceID = resolved
		}
	}

	start := time.Now()
	if err := w.run(ctx, job.Args, workspaceID, log); err != nil {
		log.Error("parse failed", "error", err, "duration_ms", time.Since(start).Milliseconds())
		if markErr := w.store.MarkError(ctx, job.Args.TaskID, asynctask.ErrorTypeGeneric, err.Error()); markErr != nil {
			log.Error("mark error failed", "error", markErr)
		}
		return err
	}

	if err := w.store.MarkSuccess(ctx, job.Args.TaskID, time.Since(start).Milliseconds()); err != nil {
		log.Error("mark success failed", "error", err)
		return err
	}
	log.Info("parse succeeded", "duration_ms", time.Since(start).Milliseconds())
	return nil
}

func (w *ParseFileToChunksWorker) run(ctx context.Context, args ParseFileToChunksArgs, workspaceID string, log *slog.Logger) error {
	// 1. Fetch file metadata from public.files
	fileName, fileType, fileURL, err := w.fetchFileMeta(ctx, args.FileID, args.UserID)
	if err != nil {
		return fmt.Errorf("fetch file meta: %w", err)
	}
	log.Info("file metadata fetched", "name", fileName, "type", fileType, "url_len", len(fileURL))

	// 2. Download file bytes
	content, err := w.fetcher.Fetch(ctx, fileURL)
	if err != nil {
		return fmt.Errorf("fetch content: %w", err)
	}
	if len(content) == 0 {
		return fmt.Errorf("empty file content")
	}
	log.Info("content fetched", "bytes", len(content))

	// 3. Run chunker
	chunks, err := w.chunker.Chunk(ctx, ChunkInput{
		Filename: fileName,
		FileType: fileType,
		Content:  content,
	})
	if err != nil {
		return fmt.Errorf("chunk: %w", err)
	}
	if len(chunks) == 0 {
		// Match TS: throw NoChunkError
		return fmt.Errorf("no chunk found; chunker returned empty result for %s", fileName)
	}
	log.Info("chunks produced", "count", len(chunks))

	// 4. Bulk insert chunks + file_chunks
	if err := w.persistChunks(ctx, args, workspaceID, chunks, log); err != nil {
		return fmt.Errorf("persist chunks: %w", err)
	}
	log.Info("chunks persisted")

	return nil
}

func (w *ParseFileToChunksWorker) fetchFileMeta(ctx context.Context, fileID, userID string) (name, fileType, url string, err error) {
	err = w.pool.QueryRow(ctx,
		`SELECT name, file_type, url FROM public.files WHERE id = $1 AND user_id = $2`,
		fileID, userID,
	).Scan(&name, &fileType, &url)
	if err != nil {
		if pgx.ErrNoRows == err {
			return "", "", "", fmt.Errorf("file %s not found for user %s", fileID, userID)
		}
		return "", "", "", fmt.Errorf("fetch file meta: %w", err)
	}
	return name, fileType, url, nil
}

func (w *ParseFileToChunksWorker) persistChunks(ctx context.Context, args ParseFileToChunksArgs, workspaceID string, chunks []ChunkRow, log *slog.Logger) error {
	batch := &pgx.Batch{}
	for i, c := range chunks {
		batch.Queue(`
			INSERT INTO public.chunks (text, metadata, "index", type, user_id, workspace_id)
			VALUES ($1, $2, $3, $4, $5, NULLIF($6, ''))
			RETURNING id::text`,
			c.Text, c.Metadata, i, firstNonEmpty(c.Type, "DocumentChunk"), args.UserID, workspaceID)
	}
	br := w.pool.SendBatch(ctx, batch)
	defer br.Close()

	chunkIDs := make([]string, len(chunks))
	for i := range chunks {
		if err := br.QueryRow().Scan(&chunkIDs[i]); err != nil {
			return fmt.Errorf("insert chunk %d: %w", i, err)
		}
	}

	// Bulk insert file_chunks junction rows in a second batch.
	batch2 := &pgx.Batch{}
	for _, cid := range chunkIDs {
		batch2.Queue(`
			INSERT INTO public.file_chunks (chunk_id, file_id, user_id, workspace_id)
			VALUES ($1, $2, $3, NULLIF($4, ''))
			ON CONFLICT DO NOTHING`,
			cid, args.FileID, args.UserID, workspaceID)
	}
	br2 := w.pool.SendBatch(ctx, batch2)
	defer br2.Close()
	for range chunkIDs {
		if _, err := br2.Exec(); err != nil {
			return fmt.Errorf("insert file_chunk: %w", err)
		}
	}
	return nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

// Command egent-jobs is the River worker process for the LobeHub async_tasks
// surface. It replaces the Next.js BFF polling loops in
// lobehub/apps/server/src/routers/async/{file,image,video}.ts with a single
// long-lived Go binary that registers workers against a Postgres-backed River
// queue and drains them.
//
// The first worker shipped is EmbedFileChunksWorker (the highest-frequency
// path — it fires on every file uploaded to a knowledge base). The image and
// video workers will be added in follow-up iterations; their TS handlers can
// keep running in parallel until the Go equivalents ship.
//
// Configuration:
//
//	KAWAI_PG_DSN               — Supabase DSN to the kawai database (required)
//	OPENAI_API_KEY / MODEL_API_KEY — embedder bearer token (required for file_ingest)
//	OPENAI_EMBEDDINGS_URL      — OpenAI-compatible base URL (default https://api.openai.com/v1)
//	OPENAI_EMBEDDINGS_MODEL    — model id (default text-embedding-3-small)
//	EGENT_JOBS_PORT            — /healthz HTTP port (default 10540)
//	EMBEDDING_BATCH_SIZE       — chunks per embed call (default 10)
//	EMBEDDING_CONCURRENCY      — concurrent embed calls per worker job (default 3)
//	EGENT_JOBS_MAX_WORKERS     — River queue MaxWorkers for file_ingest (default 10)
//	EGENT_JOBS_MIGRATE         — "1"/"true" runs River schema migrations on start (default true)
//	RAG_EVAL_MODEL             — chat model for RAG eval answer generation (default gpt-4o-mini)
//	RAG_EVAL_BASE_URL          — OpenAI-compatible base URL for RAG eval (default MODEL_BASE_URL)
//	RAG_EVAL_API_KEY           — bearer token for RAG eval (default MODEL_API_KEY)
//
// The binary does NOT expose the producer Insert API yet. Producers (the BFF)
// enqueue jobs by talking to River directly over the same DATABASE_URL — that
// is the recommended River pattern (one queue, many producers, one worker).
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/joho/godotenv"
	"github.com/riverqueue/river"

	"egent-jobs/asynctask"
	"egent-jobs/embeddings"
	"egent-jobs/fileingest"
	"egent-jobs/mediagen"
	"egent-jobs/memoryingest"
	"egent-jobs/queue"
	"egent-jobs/rageval"
)

var (
	version = "dev"
	healthy atomic.Bool
)

func main() {
	showVersion := flag.Bool("version", false, "print version and exit")
	flag.Parse()

	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	})))
	log := slog.Default()

	if *showVersion {
		fmt.Printf("egent-jobs %s\n", version)
		os.Exit(0)
	}

	// Load .env from CWD and any config dir passed via EGENT_JOBS_ENV_DIR.
	_ = godotenv.Load()
	if dir := os.Getenv("EGENT_JOBS_ENV_DIR"); dir != "" {
		_ = godotenv.Load(dir + "/.env")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	dsn := os.Getenv("KAWAI_PG_DSN")
	if dsn == "" {
		log.Error("KAWAI_PG_DSN is required")
		os.Exit(1)
	}
	log.Info("egent-jobs starting", "version", version, "dsn", queue.SanitizeDSN(dsn))

	// Wire the embedder. If the API key is missing we still boot so the
	// process can serve health checks and other workers (once shipped) — the
	// embedding worker will fail at runtime with a clear error.
	embedder, err := embeddings.NewOpenAIEmbedderFromEnv()
	if err != nil {
		log.Warn("embedder not configured; embedding worker will fail at runtime", "error", err)
		embedder = nil
	}

	// Build the River workers bundle. Each worker is constructed once and
	// shared across all jobs of its kind.
	workers := river.NewWorkers()

	maxWorkers := envInt("EGENT_JOBS_MAX_WORKERS", 10)
	batchSize := envInt("EMBEDDING_BATCH_SIZE", 10)
	concurrency := envInt("EMBEDDING_CONCURRENCY", 3)

	// Open the pgx pool first so worker constructors can take it. River
	// workers are registered into the bundle, then the client is built over
	// the same pool via NewWithPool (workers must be populated before the
	// River client is constructed).
	pool, err := queue.NewPool(ctx, dsn)
	if err != nil {
		log.Error("open pool failed", "error", err)
		os.Exit(1)
	}
	defer pool.Close()

	queues := map[string]river.QueueConfig{
		asynctask.QueueFileIngest: {MaxWorkers: maxWorkers},
		// Declared up front so producers can enqueue media jobs before
		// the Go image/video workers ship. The queue simply won't drain
		// them until a worker is registered.
		asynctask.QueueMediaGen:       {MaxWorkers: 4},
		asynctask.QueueRagEval:        {MaxWorkers: 10},
		asynctask.QueueMemoryIngest:   {MaxWorkers: 2},
		asynctask.QueuePersonaRefresh: {MaxWorkers: 1},
	}

	store := asynctask.NewStore(pool)
	river.AddWorker(workers, fileingest.NewEmbedFileChunksWorker(fileingest.WorkerConfig{
		Pool:        pool,
		Store:       store,
		Embedder:    embedder,
		Logger:      log,
		BatchSize:   batchSize,
		Concurrency: concurrency,
	}))
	river.AddWorker(workers, fileingest.NewParseFileToChunksWorker(fileingest.ParseWorkerConfig{
		Pool:    pool,
		Store:   store,
		Fetcher: &fileingest.HTTPFetcher{},
		Chunker: &fileingest.TextChunker{},
		Logger:  log,
	}))
	river.AddWorker(workers, mediagen.NewImageGenerationWorker(mediagen.ImageWorkerConfig{
		Pool:   pool,
		Store:  store,
		Logger: log,
	}))
	river.AddWorker(workers, mediagen.NewVideoGenerationWorker(mediagen.VideoWorkerConfig{
		Pool:   pool,
		Store:  store,
		Logger: log,
	}))
	river.AddWorker(workers, rageval.NewRagEvalWorker(rageval.Config{
		Pool:     pool,
		Store:    store,
		Embedder: embedder,
		Logger:   log,
	}))

	llmClient := memoryingest.NewOpenAIChatClientFromEnv()
	if llmClient == nil {
		log.Warn("memory_ingest: MODEL_BASE_URL not set; memory extraction worker will fail at runtime")
	} else {
		log.Info("memory_ingest: LLM client configured", "model", os.Getenv("MEMORY_EXTRACT_MODEL"))
	}
	// Pass a clean nil interface (not a typed-nil *OpenAIEmbedder) when no
	// embedder is configured, so PgIngestStore's `embedder == nil` guard
	// holds and EmbedBatch is never called on a nil pointer.
	var ingestEmbedder embeddings.Embedder
	if embedder != nil {
		ingestEmbedder = embedder
	}
	river.AddWorker(workers, memoryingest.NewIngestWorker(memoryingest.Config{
		Pool:     pool,
		Store:    memoryingest.NewPgIngestStore(pool, ingestEmbedder),
		LLM:      llmClient,
		Embedder: ingestEmbedder,
		Logger:   log,
	}))

	personaStore := memoryingest.NewPgPersonaStore(pool)
	river.AddWorker(workers, memoryingest.NewPersonaWorker(memoryingest.PersonaConfig{
		Pool:   pool,
		Store:  personaStore,
		LLM:    llmClient,
		Logger: log,
	}))
	log.Info("workers registered",
		"kinds", []string{
			fileingest.EmbedFileChunksArgs{}.Kind(),
			fileingest.ParseFileToChunksArgs{}.Kind(),
			mediagen.GenerateImageArgs{}.Kind(),
			mediagen.GenerateVideoArgs{}.Kind(),
			rageval.RagEvalArgs{}.Kind(),
			memoryingest.MemoryIngestArgs{}.Kind(),
			memoryingest.PersonaRefreshArgs{}.Kind(),
		},
		"batch_size", batchSize,
		"concurrency", concurrency,
		"max_workers", maxWorkers,
	)

	migrate := envBool("EGENT_JOBS_MIGRATE", true)

	// Build the River client over the pool now that all workers are registered.
	// The workers bundle is a snapshot passed at construction (not live-mutable).
	q, err := queue.NewWithPool(ctx, pool, queue.Options{
		Workers: workers,
		Queues:  queues,
		Logger:  log,
	})
	if err != nil {
		log.Error("create river client failed", "error", err)
		os.Exit(1)
	}

	if err := q.Start(ctx, migrate); err != nil {
		log.Error("start river client failed", "error", err)
		os.Exit(1)
	}
	log.Info("river client started", "migrate", migrate)

	// Lightweight HTTP server for planoctl health checks.
	port := os.Getenv("EGENT_JOBS_PORT")
	if port == "" {
		port = "10540"
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		if healthy.Load() {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("ok"))
			return
		}
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("starting"))
	})
	mux.HandleFunc("/version", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprintf(w, "egent-jobs %s\n", version)
	})

	srv := &http.Server{
		Addr:         "0.0.0.0:" + port,
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
	}

	go func() {
		log.Info("http server starting", "addr", srv.Addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("http server error", "error", err)
		}
	}()

	healthy.Store(true)

	// Graceful shutdown on SIGINT/SIGTERM.
	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGTERM)
	<-done
	log.Info("shutdown signal received, draining river workers...")
	healthy.Store(false)

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Warn("http shutdown error", "error", err)
	}
	if err := q.Stop(shutdownCtx); err != nil {
		log.Error("river stop failed", "error", err)
	}
	cancel()
	log.Info("egent-jobs stopped")
}

func envInt(key string, def int) int {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		slog.Default().Warn("invalid int env, using default", "key", key, "value", v, "default", def)
		return def
	}
	return n
}

func envBool(key string, def bool) bool {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		return def
	}
	return b
}

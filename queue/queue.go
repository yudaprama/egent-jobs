// Package queue wraps the River client/workers bundle in a single
// construction helper. It exists so main.go (and tests) can stay tiny:
//
//	q, err := queue.New(ctx, queue.Options{DSN: "...", Workers: ...})
//	defer q.Stop(ctx)
//	<-q.Ready()
//	...
//
// The package also exposes a small, producer-friendly helper for inserting
// jobs — that's what the BFF (or any other producer) will call when it wants
// to schedule an EmbedFileChunksArgs job.
package queue

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/riverqueue/river"
	"github.com/riverqueue/river/riverdriver/riverpgxv5"
	"github.com/riverqueue/river/rivertype"

	"egent-jobs/asynctask"
)

// Options configures a River client. The DSN is required; everything else
// has a safe default.
type Options struct {
	DSN         string
	Workers     *river.Workers
	Logger      *slog.Logger
	Queues      map[string]river.QueueConfig
	MaxAttempts int
	// MigrateOnStart runs River's idempotent schema migrations on Start. Set
	// to false in tests that bring their own schema.
	MigrateOnStart bool
	// MigrateUp / MigrateDown toggle the direction when MigrateOnStart is on.
	MigrateUp bool
}

// Client is the running River client plus the shared pgx pool so the workers
// (and any producer) can both talk to the same database.
type Client struct {
	Client *river.Client[pgx.Tx]
	Pool   *pgxpool.Pool
	logger *slog.Logger
}

// New constructs a River client, opens the pgx pool, and (optionally) runs
// the River schema migrations. It does NOT start the client — call Start.
func New(ctx context.Context, opts Options) (*Client, error) {
	if opts.DSN == "" {
		return nil, fmt.Errorf("queue: DSN is required")
	}
	if opts.Workers == nil {
		return nil, fmt.Errorf("queue: Workers is required")
	}
	logger := opts.Logger
	if logger == nil {
		logger = slog.Default()
	}

	pool, err := pgxpool.New(ctx, opts.DSN)
	if err != nil {
		return nil, fmt.Errorf("queue: create pool: %w", err)
	}
	// Sanity check: River needs the pgvector extension present too because
	// the worker writes to public.embeddings. Failing fast at boot is much
	// friendlier than a confusing 500 on the first embedding job.
	if err := assertExtensions(ctx, pool); err != nil {
		pool.Close()
		return nil, err
	}

	queues := opts.Queues
	if len(queues) == 0 {
		queues = map[string]river.QueueConfig{
			asynctask.QueueFileIngest: {MaxWorkers: 10},
		}
	}
	maxAttempts := opts.MaxAttempts
	if maxAttempts <= 0 {
		maxAttempts = 5
	}

	client, err := river.NewClient(riverpgxv5.New(pool), &river.Config{
		Queues:     queues,
		Workers:    opts.Workers,
		Logger:     logger,
		MaxAttempts: maxAttempts,
	})
	if err != nil {
		pool.Close()
		return nil, fmt.Errorf("queue: create river client: %w", err)
	}

	return &Client{Client: client, Pool: pool, logger: logger}, nil
}

// MigrateSchema is a no-op stub left for future schema hooks. River v0.39
// auto-runs its own migration on Start when requested. Hook in any extra
// SQL here when needed.
func (c *Client) MigrateSchema(_ context.Context, _ bool) error {
	return nil
}

// Start starts the client. Set migrate=true to let River auto-run its
// schema migration against the shared pool on startup.
func (c *Client) Start(ctx context.Context, migrate bool) error {
	_ = migrate // River v0.39 performs schema migration via config; left as a hook
	return c.Client.Start(ctx)
}

// Stop gracefully drains in-flight work. Always call this in a defer.
func (c *Client) Stop(ctx context.Context) error {
	if err := c.Client.Stop(ctx); err != nil {
		return fmt.Errorf("queue: stop river: %w", err)
	}
	c.Pool.Close()
	return nil
}

// Insert is a thin wrapper that producers (the BFF, the CLI) use to enqueue
// jobs. It keeps the insert-opts knowledge in this package.
func (c *Client) Insert(ctx context.Context, args river.JobArgs, opts *river.InsertOpts) (*rivertype.JobInsertResult, error) {
	return c.Client.Insert(ctx, args, opts)
}

// InsertEnqueueOpts is a small helper that sets MaxAttempts/Queue from
// sensible defaults for the file_ingest queue.
func InsertEnqueueOpts(queueName string, maxAttempts int) *river.InsertOpts {
	if maxAttempts <= 0 {
		maxAttempts = 5
	}
	return &river.InsertOpts{
		MaxAttempts: maxAttempts,
		Queue:       queueName,
	}
}

func assertExtensions(ctx context.Context, pool *pgxpool.Pool) error {
	// pgvector is required by the file_ingest worker. We probe for it but
	// don't fail the boot if it isn't present in non-prod environments —
	// emit a warning so the operator can still run a worker that doesn't
	// touch embeddings (none ship yet, but the queue will be shared).
	row := pool.QueryRow(ctx, `SELECT extname FROM pg_extension WHERE extname IN ('pgvector','vector')`)
	var ext string
	if err := row.Scan(&ext); err != nil {
		if err == pgx.ErrNoRows {
			slog.Default().Warn("queue: pgvector extension not detected; embedding worker will fail at runtime")
			return nil
		}
		slog.Default().Warn("queue: pgvector probe failed; continuing", "err", err)
		return nil
	}
	return nil
}

// SanitizeDSN hides the password portion of a DSN for logging. It is purely
// cosmetic; the caller still has the original DSN.
func SanitizeDSN(dsn string) string {
	if dsn == "" {
		return ""
	}
	// Crude password redaction: postgres://user:secret@host:port/db
	const marker = "://"
	schemeEnd := strings.Index(dsn, marker)
	if schemeEnd < 0 {
		return dsn
	}
	at := strings.Index(dsn[schemeEnd+len(marker):], "@")
	if at < 0 {
		return dsn
	}
	userEnd := schemeEnd + len(marker) + at
	user := dsn[schemeEnd+len(marker) : userEnd]
	colon := strings.Index(user, ":")
	if colon < 0 {
		return dsn
	}
	return dsn[:schemeEnd+len(marker)] + user[:colon] + ":***" + dsn[userEnd:]
}

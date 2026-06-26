package memoryingest

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/riverqueue/river"

	"egent-jobs/embeddings"
)

// QueueMemoryIngest is the River queue name for memory extraction jobs.
// Registered in queue.Options.Queues when the worker is wired.
const QueueMemoryIngest = "memory_ingest"

// MemoryIngestArgs is the River job payload. The BFF producer inserts
// one job per chat topic that the user selects for memory extraction.
type MemoryIngestArgs struct {
	UserID   string `json:"userId"`
	TopicID  string `json:"topicId"`
	Content  string `json:"content"` // the concatenated user messages from the topic
	Source   string `json:"source"`  // "chat_topic" (matches LobeHub MemorySourceType)
}

func (MemoryIngestArgs) Kind() string { return "memory_ingest" }

// IngestWorker processes memory extraction jobs. Each job:
//  1. Calls the LLM gatekeeper to decide whether the topic is worth extracting.
//  2. For each layer (identity, activity, context, experience, preference),
//     calls the LLM extractor with a layer-specific prompt.
//  3. Writes extracted memories through the palace Store (parent + child in tx).
//
// The worker follows the same pattern as rageval/rageval.go:
// Config struct → NewIngestWorker → river.WorkerDefaults[Args] → Work.
type IngestWorker struct {
	river.WorkerDefaults[MemoryIngestArgs]

	pool    *pgxpool.Pool
	store   Store // palace.Store interface (import cycle broken via local interface)
	llm     LLMClient
	embedder embeddings.Embedder
	client  *http.Client
	logger  *slog.Logger
	timeout time.Duration
}

// Store is the subset of palace.Store that the extraction worker
// needs. Using a local interface avoids an import cycle between
// egent-jobs and egent-lobehub/memory/palace.
type Store interface {
	CreateIdentity(ctx context.Context, userID string, in CreateIdentityInput) (string, error)
	CreateActivity(ctx context.Context, userID string, in CreateActivityInput) (string, error)
	CreateContext(ctx context.Context, userID string, in CreateContextInput) (string, error)
	CreateExperience(ctx context.Context, userID string, in CreateExperienceInput) (string, error)
	CreatePreference(ctx context.Context, userID string, in CreatePreferenceInput) (string, error)
}

// CreateIdentityInput mirrors palace.IdentityInput for decoupling.
type CreateIdentityInput struct {
	Description  string  `json:"description"`
	Type         string  `json:"type,omitempty"`
	Role         string  `json:"role,omitempty"`
	Relationship string  `json:"relationship,omitempty"`
	EpisodicDate *string `json:"episodicDate,omitempty"`
	Tags         []string `json:"tags,omitempty"`
}

// CreateActivityInput mirrors palace.ActivityInput for decoupling.
type CreateActivityInput struct {
	Type        string  `json:"type"`
	Narrative   string  `json:"narrative,omitempty"`
	Notes       string  `json:"notes,omitempty"`
	Status      string  `json:"status,omitempty"`
	Tags        []string `json:"tags,omitempty"`
}

// CreateContextInput mirrors palace.ContextInput for decoupling.
type CreateContextInput struct {
	Title         string   `json:"title"`
	Description   string   `json:"description,omitempty"`
	Type          string   `json:"type,omitempty"`
	Tags          []string `json:"tags,omitempty"`
}

// CreateExperienceInput mirrors palace.ExperienceInput for decoupling.
type CreateExperienceInput struct {
	Situation   string  `json:"situation,omitempty"`
	Action      string  `json:"action,omitempty"`
	KeyLearning string  `json:"keyLearning,omitempty"`
	Type        string  `json:"type,omitempty"`
	Tags        []string `json:"tags,omitempty"`
}

// CreatePreferenceInput mirrors palace.PreferenceInput for decoupling.
type CreatePreferenceInput struct {
	Suggestions string  `json:"suggestions,omitempty"`
	Type        string  `json:"type,omitempty"`
	Tags        []string `json:"tags,omitempty"`
}

// LLMClient abstracts the LLM call so the worker can be tested
// with a fake. In production this wraps the OpenAI-compatible chat
// completions endpoint.
type LLMClient interface {
	// Chat sends a system+user prompt and returns the assistant's reply.
	Chat(ctx context.Context, systemPrompt, userPrompt string) (string, error)
}

// Config configures the memory extraction worker.
type Config struct {
	Pool       *pgxpool.Pool
	Store      Store
	LLM        LLMClient
	Embedder   embeddings.Embedder
	HTTPClient *http.Client
	Logger     *slog.Logger
	Timeout    time.Duration
}

// NewIngestWorker creates a memory extraction worker.
func NewIngestWorker(cfg Config) *IngestWorker {
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 120 * time.Second}
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 5 * time.Minute
	}
	return &IngestWorker{
		pool:    cfg.Pool,
		store:   cfg.Store,
		llm:     cfg.LLM,
		embedder: cfg.Embedder,
		client:  cfg.HTTPClient,
		logger:  cfg.Logger,
		timeout: cfg.Timeout,
	}
}

// Work satisfies river.Worker.
func (w *IngestWorker) Work(ctx context.Context, job *river.Job[MemoryIngestArgs]) error {
	log := w.logger.With(
		"job_id", job.ID,
		"kind", job.Kind,
		"attempt", job.Attempt,
		"user_id", job.Args.UserID,
		"topic_id", job.Args.TopicID,
		"source", job.Args.Source,
	)
	ctx, cancel := context.WithTimeout(ctx, w.timeout)
	defer cancel()

	start := time.Now()
	if err := w.run(ctx, job.Args, log); err != nil {
		log.Error("memory extraction failed", "error", err, "duration_ms", time.Since(start).Milliseconds())
		return fmt.Errorf("memory extraction failed: %w", err)
	}
	log.Info("memory extraction succeeded", "duration_ms", time.Since(start).Milliseconds())
	return nil
}

// run executes the full extraction pipeline for a single topic.
func (w *IngestWorker) run(ctx context.Context, args MemoryIngestArgs, log *slog.Logger) error {
	userID := args.UserID
	content := args.Content
	if content == "" {
		log.Info("empty content, skipping extraction", "topic_id", args.TopicID)
		return nil
	}

	// --- Step 1: Gatekeeper ---
	isRelevant, err := w.gatekeeper(ctx, content, log)
	if err != nil {
		return fmt.Errorf("gatekeeper: %w", err)
	}
	if !isRelevant {
		log.Info("topic deemed not relevant for memory extraction by gatekeeper")
		return nil
	}

	totalStored := 0

	// --- Step 2: Identity ---
	if memories, err := w.extractIdentity(ctx, content, log); err != nil {
		log.Warn("identity extraction failed", "error", err)
	} else {
		for _, m := range memories {
			if _, err := w.store.CreateIdentity(ctx, userID, m); err != nil {
				log.Error("persist identity failed", "error", err)
			} else {
				totalStored++
			}
		}
	}

	// --- Step 3: Activity ---
	if memories, err := w.extractActivity(ctx, content, log); err != nil {
		log.Warn("activity extraction failed", "error", err)
	} else {
		for _, m := range memories {
			if _, err := w.store.CreateActivity(ctx, userID, m); err != nil {
				log.Error("persist activity failed", "error", err)
			} else {
				totalStored++
			}
		}
	}

	// --- Step 4: Context ---
	if memories, err := w.extractContext(ctx, content, log); err != nil {
		log.Warn("context extraction failed", "error", err)
	} else {
		for _, m := range memories {
			if _, err := w.store.CreateContext(ctx, userID, m); err != nil {
				log.Error("persist context failed", "error", err)
			} else {
				totalStored++
			}
		}
	}

	// --- Step 5: Experience ---
	if memories, err := w.extractExperience(ctx, content, log); err != nil {
		log.Warn("experience extraction failed", "error", err)
	} else {
		for _, m := range memories {
			if _, err := w.store.CreateExperience(ctx, userID, m); err != nil {
				log.Error("persist experience failed", "error", err)
			} else {
				totalStored++
			}
		}
	}

	// --- Step 6: Preference ---
	if memories, err := w.extractPreference(ctx, content, log); err != nil {
		log.Warn("preference extraction failed", "error", err)
	} else {
		for _, m := range memories {
			if _, err := w.store.CreatePreference(ctx, userID, m); err != nil {
				log.Error("persist preference failed", "error", err)
			} else {
				totalStored++
			}
		}
	}

	log.Info("extraction complete", "memories_stored", totalStored)
	return nil
}
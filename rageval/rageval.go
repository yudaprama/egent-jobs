// Package rageval implements the River worker that ports the deleted
// runRecordEvaluation handler from lobehub/apps/server/src/routers/async/ragEval.ts.
//
// Each job processes a single rag_eval_evaluation_records row:
//  1. Embed the question (if question_embedding_id is null).
//  2. Retrieve relevant chunks via pgvector cosine similarity (if context is empty).
//  3. Generate an LLM answer using the retrieved context.
//  4. Update the eval record with the answer and mark it Success.
//  5. On error, mark both the record and its parent evaluation as Error.
//
// Configuration:
//
//	MODEL_BASE_URL / RAG_EVAL_BASE_URL — OpenAI-compatible base URL (default https://api.openai.com/v1)
//	MODEL_API_KEY  / RAG_EVAL_API_KEY  — bearer token (required)
//	RAG_EVAL_MODEL — chat model id (default gpt-4o-mini)
package rageval

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/riverqueue/river"

	"egent-jobs/asynctask"
	"egent-jobs/embeddings"
)

// ---------- job args ----------

// RagEvalArgs is the River job payload. The BFF producer creates one job per
// evaluation record row in startEvaluationTask.
type RagEvalArgs struct {
	EvalRecordID string `json:"evalRecordId"`
	UserID       string `json:"userId"`
	WorkspaceID  string `json:"workspaceId,omitempty"`
}

func (RagEvalArgs) Kind() string { return "rag_eval_record_evaluation" }

// ---------- worker ----------

// RagEvalWorker processes RAG evaluation records end-to-end: embed → retrieve →
// generate → persist. It reuses the embeddings.Embedder for the embedding step
// and calls an OpenAI-compatible chat endpoint for the answer generation.
type RagEvalWorker struct {
	river.WorkerDefaults[RagEvalArgs]

	pool    *pgxpool.Pool
	store   *asynctask.Store
	embed   embeddings.Embedder
	client  *http.Client
	logger  *slog.Logger
	timeout time.Duration
}

// Config configures the RAG eval worker.
type Config struct {
	Pool      *pgxpool.Pool
	Store     *asynctask.Store
	Embedder  embeddings.Embedder
	Client    *http.Client
	Logger    *slog.Logger
	Timeout   time.Duration
}

var defaultHTTPClient = &http.Client{Timeout: 120 * time.Second}

func NewRagEvalWorker(cfg Config) *RagEvalWorker {
	if cfg.Client == nil {
		cfg.Client = defaultHTTPClient
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 5 * time.Minute
	}
	return &RagEvalWorker{
		pool:    cfg.Pool,
		store:   cfg.Store,
		embed:   cfg.Embedder,
		client:  cfg.Client,
		logger:  cfg.Logger,
		timeout: cfg.Timeout,
	}
}

// Work satisfies river.Worker.
func (w *RagEvalWorker) Work(ctx context.Context, job *river.Job[RagEvalArgs]) error {
	log := w.logger.With(
		"job_id", job.ID,
		"kind", job.Kind,
		"attempt", job.Attempt,
		"eval_record_id", job.Args.EvalRecordID,
		"user_id", job.Args.UserID,
	)
	ctx, cancel := context.WithTimeout(ctx, w.timeout)
	defer cancel()

	// Note: RAG eval doesn't use async_tasks for status tracking in the same
	// way as file_ingest/mediagen — the eval record itself has a status column.
	// We skip MarkProcessing on async_tasks here.

	start := time.Now()
	if err := w.run(ctx, job.Args, log); err != nil {
		log.Error("rag eval failed", "error", err, "duration_ms", time.Since(start).Milliseconds())
		return err
	}
	log.Info("rag eval succeeded", "duration_ms", time.Since(start).Milliseconds())
	return nil
}

// ---------- eval record types ----------

type evalRecord struct {
	ID                 string
	Question           string
	Answer             *string
	Context            []string // text[] in PG — may be nil
	Ideal              *string
	Status             string
	LanguageModel      *string
	EmbeddingModel     *string
	QuestionEmbeddingID *string
	Duration           *int
	DatasetRecordID    string
	EvaluationID       string
	UserID             string
	WorkspaceID        *string
}

type datasetRecord struct {
	ID             string
	Question       *string
	Ideal          *string
	ReferenceFiles []string // text[] in PG
}

// ---------- main flow ----------

func (w *RagEvalWorker) run(ctx context.Context, args RagEvalArgs, log *slog.Logger) error {
	rec, err := w.fetchEvalRecord(ctx, args.EvalRecordID, args.UserID)
	if err != nil {
		if err == pgx.ErrNoRows {
			log.Warn("eval record missing; cancelling")
			return river.JobCancel(fmt.Errorf("eval record %s not found", args.EvalRecordID))
		}
		return w.fail(ctx, args.EvalRecordID, "", err, log)
	}

	// Step 1: Embed the question if needed.
	embeddingID := rec.QuestionEmbeddingID
	var embeddingVec []float32

	if embeddingID == nil || *embeddingID == "" {
		vec, newID, err := w.embedQuestion(ctx, rec)
		if err != nil {
			return w.fail(ctx, rec.ID, rec.EvaluationID, fmt.Errorf("embed question: %w", err), log)
		}
		embeddingID = &newID
		embeddingVec = vec
		log.Info("question embedded", "embedding_id", newID)
	} else {
		// Fetch existing embedding vector.
		vec, err := w.fetchEmbedding(ctx, *embeddingID)
		if err != nil {
			return w.fail(ctx, rec.ID, rec.EvaluationID, fmt.Errorf("fetch embedding: %w", err), log)
		}
		embeddingVec = vec
	}

	// Step 2: Retrieve context if not already set.
	ctx_texts := rec.Context
	if len(ctx_texts) == 0 {
		dsRec, err := w.fetchDatasetRecord(ctx, rec.DatasetRecordID)
		if err != nil {
			return w.fail(ctx, rec.ID, rec.EvaluationID, fmt.Errorf("fetch dataset record: %w", err), log)
		}

		if len(dsRec.ReferenceFiles) > 0 && len(embeddingVec) > 0 {
			chunks, err := w.retrieveContext(ctx, dsRec.ReferenceFiles, args.UserID, embeddingVec)
			if err != nil {
				return w.fail(ctx, rec.ID, rec.EvaluationID, fmt.Errorf("retrieve context: %w", err), log)
			}
			ctx_texts = chunks
			if err := w.persistContext(ctx, rec.ID, ctx_texts); err != nil {
				log.Warn("persist context failed (non-fatal)", "error", err)
			}
			log.Info("context retrieved", "chunks", len(ctx_texts))
		}
	}

	// Step 3: Generate LLM answer.
	answer, err := w.generateAnswer(ctx, rec.Question, ctx_texts, rec.LanguageModel)
	if err != nil {
		return w.fail(ctx, rec.ID, rec.EvaluationID, fmt.Errorf("generate answer: %w", err), log)
	}

	// Step 4: Persist result.
	if err := w.persistResult(ctx, rec.ID, answer); err != nil {
		return w.fail(ctx, rec.ID, rec.EvaluationID, fmt.Errorf("persist result: %w", err), log)
	}

	log.Info("eval record processed", "answer_len", len(answer))
	return nil
}

// ---------- DB queries ----------

func (w *RagEvalWorker) fetchEvalRecord(ctx context.Context, evalRecordID, userID string) (*evalRecord, error) {
	var rec evalRecord
	var answer, ideal, langModel, embedModel, qEmbID *string
	var wsID *string

	err := w.pool.QueryRow(ctx, `
		SELECT id, question, answer, context, ideal, status,
		       language_model, embedding_model, question_embedding_id,
		       dataset_record_id, evaluation_id, user_id, workspace_id::text
		  FROM public.rag_eval_evaluation_records
		 WHERE id = $1 AND user_id = $2`,
		evalRecordID, userID,
	).Scan(
		&rec.ID, &rec.Question, &answer, &rec.Context, &ideal, &rec.Status,
		&langModel, &embedModel, &qEmbID,
		&rec.DatasetRecordID, &rec.EvaluationID, &rec.UserID, &wsID,
	)
	if err != nil {
		return nil, err
	}
	rec.Answer = answer
	rec.Ideal = ideal
	rec.LanguageModel = langModel
	rec.EmbeddingModel = embedModel
	rec.QuestionEmbeddingID = qEmbID
	rec.WorkspaceID = wsID
	return &rec, nil
}

func (w *RagEvalWorker) fetchDatasetRecord(ctx context.Context, id string) (*datasetRecord, error) {
	var rec datasetRecord
	var question, ideal *string

	err := w.pool.QueryRow(ctx, `
		SELECT id, question, ideal, reference_files
		  FROM public.rag_eval_dataset_records
		 WHERE id = $1`, id,
	).Scan(&rec.ID, &question, &ideal, &rec.ReferenceFiles)
	if err != nil {
		return nil, err
	}
	rec.Question = question
	rec.Ideal = ideal
	return &rec, nil
}

// embedQuestion calls the embedder for the eval record's question and persists
// the resulting vector to public.embeddings (with chunk_id = NULL since this is
// a question embedding, not a chunk embedding). Returns the vector and the new
// embedding row ID.
func (w *RagEvalWorker) embedQuestion(ctx context.Context, rec *evalRecord) ([]float32, string, error) {
	if w.embed == nil {
		return nil, "", fmt.Errorf("embedder not configured")
	}

	inputs := []embeddings.Input{{ChunkID: rec.ID, Text: rec.Question}}
	results, err := w.embed.EmbedBatch(ctx, inputs)
	if err != nil {
		return nil, "", err
	}
	if len(results) == 0 {
		return nil, "", fmt.Errorf("embedder returned 0 results")
	}

	vec := results[0].Vector
	if len(vec) != 1024 {
		return nil, "", fmt.Errorf("vector dimension mismatch: got %d want 1024", len(vec))
	}

	// Persist the embedding. chunk_id is NULL for eval question embeddings.
	var embeddingID string
	err = w.pool.QueryRow(ctx, `
		INSERT INTO public.embeddings
			(embeddings, model, user_id, workspace_id, client_id)
		VALUES ($1::vector, $2, $3, NULLIF($4, ''), NULL)
		RETURNING id`,
		vectorToPG(vec), w.embed.Model(), rec.UserID, deref(rec.WorkspaceID),
	).Scan(&embeddingID)
	if err != nil {
		return nil, "", fmt.Errorf("insert embedding: %w", err)
	}

	// Update the eval record with the new embedding ID.
	_, err = w.pool.Exec(ctx, `
		UPDATE public.rag_eval_evaluation_records
		   SET question_embedding_id = $2, updated_at = NOW()
		 WHERE id = $1`,
		rec.ID, embeddingID)
	if err != nil {
		return nil, "", fmt.Errorf("update eval record embedding_id: %w", err)
	}

	return vec, embeddingID, nil
}

// fetchEmbedding reads an existing embedding vector by ID.
func (w *RagEvalWorker) fetchEmbedding(ctx context.Context, embeddingID string) ([]float32, error) {
	var vecStr string
	err := w.pool.QueryRow(ctx, `
		SELECT embeddings::text
		  FROM public.embeddings
		 WHERE id = $1`, embeddingID,
	).Scan(&vecStr)
	if err != nil {
		return nil, fmt.Errorf("fetch embedding %s: %w", embeddingID, err)
	}
	return parsePGVector(vecStr), nil
}

// retrieveContext performs a semantic search against chunks that belong to the
// given reference file IDs. Returns up to 15 chunk texts ordered by cosine
// similarity descending.
func (w *RagEvalWorker) retrieveContext(ctx context.Context, fileIDs []string, userID string, queryVec []float32) ([]string, error) {
	rows, err := w.pool.Query(ctx, `
		SELECT c.text
		  FROM public.chunks c
		  JOIN public.embeddings e ON e.chunk_id = c.id
		  JOIN public.file_chunks fc ON fc.chunk_id = c.id
		 WHERE fc.file_id = ANY($1)
		   AND c.user_id = $2
		 ORDER BY 1 - (e.embeddings <=> $3::vector) DESC
		 LIMIT 15`,
		fileIDs, userID, vectorToPG(queryVec),
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var texts []string
	for rows.Next() {
		var t string
		if err := rows.Scan(&t); err != nil {
			return nil, err
		}
		if t != "" {
			texts = append(texts, t)
		}
	}
	return texts, rows.Err()
}

func (w *RagEvalWorker) persistContext(ctx context.Context, evalRecordID string, context []string) error {
	_, err := w.pool.Exec(ctx, `
		UPDATE public.rag_eval_evaluation_records
		   SET context = $2, updated_at = NOW()
		 WHERE id = $1`,
		evalRecordID, context)
	return err
}

// ---------- LLM chat ----------

func (w *RagEvalWorker) generateAnswer(ctx context.Context, question string, context []string, languageModel *string) (string, error) {
	baseURL := ragEvalBaseURL()
	apiKey := ragEvalAPIKey()
	model := ragEvalModel()
	if languageModel != nil && *languageModel != "" {
		model = *languageModel
	}

	systemPrompt := buildSystemPrompt(context)

	body := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "system", "content": systemPrompt},
			{"role": "user", "content": question},
		},
		"temperature": 1,
		"stream":      false,
	}

	raw, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("marshal chat request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/chat/completions", bytes.NewReader(raw))
	if err != nil {
		return "", fmt.Errorf("build chat request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := w.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("chat call: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read chat response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("chat returned %d: %s", resp.StatusCode, truncate(string(respBody), 500))
	}

	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return "", fmt.Errorf("decode chat response: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return "", fmt.Errorf("chat returned 0 choices")
	}
	return parsed.Choices[0].Message.Content, nil
}

func buildSystemPrompt(context []string) string {
	if len(context) == 0 {
		return "Answer the question based on your knowledge."
	}
	var b strings.Builder
	b.WriteString("Answer the question based on the following context:\n\n")
	for i, c := range context {
		fmt.Fprintf(&b, "[Context %d]\n%s\n\n", i+1, c)
	}
	return b.String()
}

// ---------- persist / fail ----------

func (w *RagEvalWorker) persistResult(ctx context.Context, evalRecordID, answer string) error {
	_, err := w.pool.Exec(ctx, `
		UPDATE public.rag_eval_evaluation_records
		   SET answer = $2,
		       status = 'Success',
		       duration = EXTRACT(EPOCH FROM (NOW() - created_at)) * 1000,
		       updated_at = NOW()
		 WHERE id = $1`,
		evalRecordID, answer)
	return err
}

func (w *RagEvalWorker) fail(ctx context.Context, evalRecordID, evaluationID string, err error, log *slog.Logger) error {
	log.Error("rag eval failed", "error", err)

	errPayload := fmt.Sprintf(`{"name":"RagEvalError","message":%q}`, err.Error())

	// Mark the eval record as Error.
	_, markErr := w.pool.Exec(ctx, `
		UPDATE public.rag_eval_evaluation_records
		   SET status = 'Error', error = $2::jsonb, updated_at = NOW()
		 WHERE id = $1`,
		evalRecordID, errPayload)
	if markErr != nil {
		log.Error("mark eval record error failed", "error", markErr)
	}

	// Mark the parent evaluation as Error.
	if evaluationID != "" {
		_, markErr := w.pool.Exec(ctx, `
			UPDATE public.rag_eval_evaluations
			   SET status = 'Error', updated_at = NOW()
			 WHERE id = $1`, evaluationID)
		if markErr != nil {
			log.Error("mark evaluation error failed", "error", markErr)
		}
	}

	return err
}

// ---------- env helpers ----------

func ragEvalBaseURL() string {
	if v := os.Getenv("RAG_EVAL_BASE_URL"); v != "" {
		return strings.TrimRight(v, "/")
	}
	if v := os.Getenv("MODEL_BASE_URL"); v != "" {
		return strings.TrimRight(v, "/")
	}
	return "https://api.openai.com/v1"
}

func ragEvalAPIKey() string {
	if v := os.Getenv("RAG_EVAL_API_KEY"); v != "" {
		return v
	}
	return os.Getenv("MODEL_API_KEY")
}

func ragEvalModel() string {
	if v := os.Getenv("RAG_EVAL_MODEL"); v != "" {
		return v
	}
	return "gpt-4o-mini"
}

// ---------- utilities ----------

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func vectorToPG(v []float32) string {
	b := make([]byte, 0, 1+len(v)*8)
	b = append(b, '[')
	for i, x := range v {
		if i > 0 {
			b = append(b, ',')
		}
		b = fmt.Appendf(b, "%g", x)
	}
	b = append(b, ']')
	return string(b)
}

// parsePGVector parses a pgvector text representation like "[1.0,2.0,3.0]"
// into a float32 slice.
func parsePGVector(s string) []float32 {
	s = strings.TrimPrefix(s, "[")
	s = strings.TrimSuffix(s, "]")
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]float32, 0, len(parts))
	for _, p := range parts {
		var f float64
		fmt.Sscanf(strings.TrimSpace(p), "%g", &f)
		out = append(out, float32(f))
	}
	return out
}

// Interface compliance.
var _ river.Worker[RagEvalArgs] = (*RagEvalWorker)(nil)

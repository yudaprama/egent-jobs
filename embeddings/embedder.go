// Package embeddings defines the embedder contract the file-ingest workers
// depend on. It mirrors the shape of modelRuntime.embeddings() in
// lobehub/apps/server/src/routers/async/file.ts:131, but as an interface so
// the worker can be tested with a fake and so the OpenAI-compatible
// implementation can be swapped without touching the worker.
package embeddings

import "context"

// Result is a single embedding vector paired with the chunk it was derived
// from. The ChunkID comes from the chunks table; the Vector is persisted to
// public.embeddings which is constrained to vector(1024).
type Result struct {
	ChunkID string
	Vector  []float32
}

// Input is a single text + the chunk it belongs to.
type Input struct {
	ChunkID string
	Text    string
}

// Embedder produces 1024-dimensional vectors for a batch of inputs in one
// round-trip. The batch size is enforced by the caller (EMBEDDING_BATCH_SIZE)
// and the dimensionality is fixed by the public.embeddings schema.
type Embedder interface {
	// EmbedBatch returns one Result per input, in order. The model name is
	// configured on the implementation (e.g. "text-embedding-3-small") so the
	// worker can stay model-agnostic.
	EmbedBatch(ctx context.Context, inputs []Input) ([]Result, error)

	// Model returns the persisted model identifier — written to
	// public.embeddings.model so the BFF can show which model produced a row.
	Model() string
}

// Package asynctask mirrors the LobeHub AsyncTaskStatus / AsyncTaskErrorType
// enums defined in lobehub/packages/types/asyncTask. The string values MUST
// match what the Next.js BFF writes to the async_tasks.status column so the
// River workers can stay in lock-step with the legacy polling code.
package asynctask

// Status values mirror AsyncTaskStatus in lobehub/packages/types/asyncTask.
const (
	StatusPending    = "pending"
	StatusProcessing = "processing"
	StatusSuccess    = "success"
	StatusError      = "error"
)

// ErrorType values mirror AsyncTaskErrorType in lobehub/packages/types/asyncTask.
const (
	ErrorTypeGeneric         = "AsyncTaskError"
	ErrorTypeTimeout         = "Timeout"
	ErrorTypeEmbeddingError  = "EmbeddingError"
	ErrorTypeContentPolicy   = "ContentPolicyError"
)

// Error is the JSONB payload persisted to async_tasks.error. Its shape must
// match the TS AsyncTaskError class so the BFF can round-trip it unchanged.
type Error struct {
	Name    string `json:"name"`
	Message string `json:"message"`
}

// Queue names. River uses opaque strings; we segregate the highest-frequency
// file ingestion path onto its own queue so a backlog of embeddings cannot
// starve image/video work (which run on separate workers in later iterations).
const (
	QueueFileIngest   = "file_ingest"
	QueueMediaGen     = "media_gen"
	QueueRagEval      = "rag_eval"
	QueueMemoryIngest = "memory_ingest"
	QueuePersonaRefresh = "persona_refresh"
)

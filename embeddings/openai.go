// Package embeddings — OpenAI-compatible HTTP embedder.
//
// This implementation hits any OpenAI-compatible /v1/embeddings endpoint
// (OpenAI, Azure OpenAI, OpenRouter, local Ollama with the OpenAI shim, etc).
// Configuration is via env vars so it mirrors the ModelRuntime wiring used by
// the Next.js BFF:
//
//	OPENAI_EMBEDDINGS_URL (or MODEL_BASE_URL) — base URL, default https://api.openai.com/v1
//	OPENAI_EMBEDDINGS_MODEL                 — model id, default text-embedding-3-small
//	OPENAI_API_KEY (or MODEL_API_KEY)        — bearer token
package embeddings

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// OpenAIEmbedder posts text batches to an OpenAI-compatible embeddings
// endpoint. It is safe for concurrent use; the underlying http.Client has no
// mutable state.
type OpenAIEmbedder struct {
	baseURL string
	model   string
	apiKey  string
	client  *http.Client
}

// OpenAIEmbedderOption applies optional configuration.
type OpenAIEmbedderOption func(*OpenAIEmbedder)

// WithHTTPClient overrides the default 30s http.Client.
func WithHTTPClient(c *http.Client) OpenAIEmbedderOption {
	return func(e *OpenAIEmbedder) { e.client = c }
}

// NewOpenAIEmbedderFromEnv reads OPENAI_EMBEDDINGS_URL / OPENAI_EMBEDDINGS_MODEL
// / OPENAI_API_KEY (falling back to MODEL_BASE_URL / MODEL_API_KEY). It returns
// an error only when the API key is missing — callers that want to defer
// configuration (e.g. for tests) can construct the struct directly.
func NewOpenAIEmbedderFromEnv() (*OpenAIEmbedder, error) {
	apiKey := firstNonEmpty(os.Getenv("OPENAI_API_KEY"), os.Getenv("MODEL_API_KEY"))
	if apiKey == "" {
		return nil, fmt.Errorf("embeddings: OPENAI_API_KEY (or MODEL_API_KEY) is required")
	}
	baseURL := firstNonEmpty(os.Getenv("OPENAI_EMBEDDINGS_URL"), os.Getenv("MODEL_BASE_URL"), "https://api.openai.com/v1")
	model := os.Getenv("OPENAI_EMBEDDINGS_MODEL")
	if model == "" {
		model = "text-embedding-3-small"
	}
	return &OpenAIEmbedder{
		baseURL: strings.TrimRight(baseURL, "/"),
		model:   model,
		apiKey:  apiKey,
		client:  &http.Client{Timeout: 60 * time.Second},
	}, nil
}

// Model implements Embedder.
func (e *OpenAIEmbedder) Model() string { return e.model }

// EmbedBatch posts one request to /v1/embeddings with dimensions=1024 so the
// resulting vectors fit the public.embeddings vector(1024) column.
func (e *OpenAIEmbedder) EmbedBatch(ctx context.Context, inputs []Input) ([]Result, error) {
	if len(inputs) == 0 {
		return nil, nil
	}
	texts := make([]string, len(inputs))
	for i, in := range inputs {
		texts[i] = in.Text
	}

	body := map[string]any{
		"input":      texts,
		"model":      e.model,
		"dimensions": 1024,
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("embeddings: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/embeddings", bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("embeddings: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embeddings: call %s: %w", e.baseURL, err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("embeddings: read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("embeddings: %s returned %d: %s", e.baseURL, resp.StatusCode, truncate(string(respBody), 500))
	}

	var parsed struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, fmt.Errorf("embeddings: decode response: %w", err)
	}
	if len(parsed.Data) != len(inputs) {
		return nil, fmt.Errorf("embeddings: expected %d vectors, got %d", len(inputs), len(parsed.Data))
	}
	out := make([]Result, len(inputs))
	for i, item := range parsed.Data {
		out[i] = Result{ChunkID: inputs[i].ChunkID, Vector: item.Embedding}
	}
	return out, nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

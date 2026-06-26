package rageval

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// ---------- Args Kind ----------

func TestRagEvalArgs_Kind(t *testing.T) {
	if got := (RagEvalArgs{}).Kind(); got != "rag_eval_record_evaluation" {
		t.Errorf("kind = %q, want %q", got, "rag_eval_record_evaluation")
	}
}

// ---------- config defaults ----------

func TestDefaultConfig(t *testing.T) {
	w := NewRagEvalWorker(Config{})
	if w.timeout != 5*time.Minute {
		t.Errorf("default timeout = %v, want 5m", w.timeout)
	}
	if w.client == nil {
		t.Error("expected default HTTP client")
	}
	if w.logger == nil {
		t.Error("expected default logger")
	}
}

// ---------- env helpers ----------

func TestEnvHelpers(t *testing.T) {
	t.Run("ragEvalBaseURL", func(t *testing.T) {
		t.Setenv("RAG_EVAL_BASE_URL", "http://test.local/v1")
		if got := ragEvalBaseURL(); got != "http://test.local/v1" {
			t.Errorf("ragEvalBaseURL = %q", got)
		}
	})
	t.Run("ragEvalBaseURL_fallback", func(t *testing.T) {
		t.Setenv("RAG_EVAL_BASE_URL", "")
		t.Setenv("MODEL_BASE_URL", "http://fallback.local/v1")
		if got := ragEvalBaseURL(); got != "http://fallback.local/v1" {
			t.Errorf("ragEvalBaseURL fallback = %q", got)
		}
	})
	t.Run("ragEvalAPIKey", func(t *testing.T) {
		t.Setenv("RAG_EVAL_API_KEY", "key1")
		if got := ragEvalAPIKey(); got != "key1" {
			t.Errorf("ragEvalAPIKey = %q", got)
		}
	})
	t.Run("ragEvalAPIKey_fallback", func(t *testing.T) {
		t.Setenv("RAG_EVAL_API_KEY", "")
		t.Setenv("MODEL_API_KEY", "fallback-key")
		if got := ragEvalAPIKey(); got != "fallback-key" {
			t.Errorf("ragEvalAPIKey fallback = %q", got)
		}
	})
	t.Run("ragEvalModel", func(t *testing.T) {
		t.Setenv("RAG_EVAL_MODEL", "gpt-4o")
		if got := ragEvalModel(); got != "gpt-4o" {
			t.Errorf("ragEvalModel = %q", got)
		}
	})
	t.Run("ragEvalModel_default", func(t *testing.T) {
		t.Setenv("RAG_EVAL_MODEL", "")
		if got := ragEvalModel(); got != "gpt-4o-mini" {
			t.Errorf("ragEvalModel default = %q", got)
		}
	})
}

// ---------- utility tests ----------

func TestBuildSystemPrompt_Empty(t *testing.T) {
	prompt := buildSystemPrompt(nil)
	if prompt != "Answer the question based on your knowledge." {
		t.Errorf("unexpected prompt: %q", prompt)
	}
}

func TestBuildSystemPrompt_WithContext(t *testing.T) {
	prompt := buildSystemPrompt([]string{"chunk A", "chunk B"})
	if !strings.Contains(prompt, "[Context 1]") {
		t.Error("missing [Context 1]")
	}
	if !strings.Contains(prompt, "chunk A") {
		t.Error("missing chunk A")
	}
	if !strings.Contains(prompt, "[Context 2]") {
		t.Error("missing [Context 2]")
	}
	if !strings.Contains(prompt, "chunk B") {
		t.Error("missing chunk B")
	}
}

func TestTruncate(t *testing.T) {
	if got := truncate("hello", 10); got != "hello" {
		t.Errorf("short: got %q", got)
	}
	if got := truncate("abcdefghijklmnop", 5); got != "abcde..." {
		t.Errorf("long: got %q", got)
	}
	if got := truncate("", 5); got != "" {
		t.Errorf("empty: got %q", got)
	}
}

func TestDeref(t *testing.T) {
	if got := deref(nil); got != "" {
		t.Errorf("nil: got %q", got)
	}
	s := "hello"
	if got := deref(&s); got != "hello" {
		t.Errorf("value: got %q", got)
	}
}

func TestVectorToPG(t *testing.T) {
	vec := []float32{1.0, 2.5, -3.14}
	got := vectorToPG(vec)
	if !strings.HasPrefix(got, "[") || !strings.HasSuffix(got, "]") {
		t.Errorf("format: %q", got)
	}
	if !strings.Contains(got, "2.5") {
		t.Errorf("missing 2.5 in %q", got)
	}
}

func TestVectorToPG_Empty(t *testing.T) {
	got := vectorToPG(nil)
	if got != "[]" {
		t.Errorf("empty vector: got %q", got)
	}
}

func TestParsePGVector(t *testing.T) {
	vec := parsePGVector("[1.0,2.5,-3.14]")
	if len(vec) != 3 {
		t.Fatalf("len = %d, want 3", len(vec))
	}
	if vec[0] != 1.0 {
		t.Errorf("vec[0] = %f, want 1.0", vec[0])
	}
	if vec[1] != 2.5 {
		t.Errorf("vec[1] = %f, want 2.5", vec[1])
	}
}

func TestParsePGVector_Empty(t *testing.T) {
	vec := parsePGVector("[]")
	if vec != nil {
		t.Errorf("empty: got %v", vec)
	}
}

// ---------- generateAnswer (mock HTTP) ----------

func TestGenerateAnswer_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}

		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)

		if body["model"] != "gpt-4o-mini" {
			t.Errorf("model = %v", body["model"])
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]string{
						"content": "The answer is 42.",
					},
				},
			},
		})
	}))
	defer srv.Close()

	t.Setenv("RAG_EVAL_BASE_URL", srv.URL)

	w := NewRagEvalWorker(Config{Client: srv.Client()})
	answer, err := w.generateAnswer(context.Background(), "What is the meaning of life?", []string{"context paragraph"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if answer != "The answer is 42." {
		t.Errorf("answer = %q", answer)
	}
}

func TestGenerateAnswer_ModelOverride(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)

		if body["model"] != "gpt-4o" {
			t.Errorf("model = %v, want gpt-4o", body["model"])
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{"content": "answer"}},
			},
		})
	}))
	defer srv.Close()

	t.Setenv("RAG_EVAL_BASE_URL", srv.URL)

	w := NewRagEvalWorker(Config{Client: srv.Client()})
	model := "gpt-4o"
	_, err := w.generateAnswer(context.Background(), "test", nil, &model)
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateAnswer_ProviderError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("internal error"))
	}))
	defer srv.Close()

	t.Setenv("RAG_EVAL_BASE_URL", srv.URL)

	w := NewRagEvalWorker(Config{Client: srv.Client()})
	_, err := w.generateAnswer(context.Background(), "test", nil, nil)
	if err == nil || !strings.Contains(err.Error(), "500") {
		t.Errorf("expected 500 error, got %v", err)
	}
}

func TestGenerateAnswer_NoChoices(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{},
		})
	}))
	defer srv.Close()

	t.Setenv("RAG_EVAL_BASE_URL", srv.URL)

	w := NewRagEvalWorker(Config{Client: srv.Client()})
	_, err := w.generateAnswer(context.Background(), "test", nil, nil)
	if err == nil || !strings.Contains(err.Error(), "0 choices") {
		t.Errorf("expected '0 choices' error, got %v", err)
	}
}

func TestGenerateAnswer_InvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("not json"))
	}))
	defer srv.Close()

	t.Setenv("RAG_EVAL_BASE_URL", srv.URL)

	w := NewRagEvalWorker(Config{Client: srv.Client()})
	_, err := w.generateAnswer(context.Background(), "test", nil, nil)
	if err == nil || !strings.Contains(err.Error(), "decode") {
		t.Errorf("expected decode error, got %v", err)
	}
}

func TestGenerateAnswer_SystemPrompt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)

		if len(body.Messages) != 2 {
			t.Errorf("expected 2 messages, got %d", len(body.Messages))
		}
		if body.Messages[0].Role != "system" {
			t.Errorf("first message role = %q", body.Messages[0].Role)
		}
		if !strings.Contains(body.Messages[0].Content, "context paragraph") {
			t.Error("system prompt missing context")
		}
		if body.Messages[1].Role != "user" {
			t.Errorf("second message role = %q", body.Messages[1].Role)
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{"content": "ok"}},
			},
		})
	}))
	defer srv.Close()

	t.Setenv("RAG_EVAL_BASE_URL", srv.URL)

	w := NewRagEvalWorker(Config{Client: srv.Client()})
	_, err := w.generateAnswer(context.Background(), "question?", []string{"context paragraph"}, nil)
	if err != nil {
		t.Fatal(err)
	}
}

// ---------- interface compliance ----------

func TestInterfaceCompliance(t *testing.T) {
	// Compile-time check via var declaration in source file.
	// This test verifies the package compiles.
	_ = NewRagEvalWorker(Config{})
}

// ---------- Work integration stub ----------

func TestRagEvalWorker_Work_RequiresPool(t *testing.T) {
	// Full Work flow requires a pgx pool. The individual steps
	// (generateAnswer, buildSystemPrompt, utility functions) are
	// tested above. Integration tests with a real database should
	// be added when a test DB is available.
	t.Skip("requires pgx pool for DB queries; unit tests cover non-DB logic")
}

package memoryingest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"sync"
	"testing"
	"time"
)

// ExtractionTrace captures the full extraction flow for debugging and parity testing.
// This is used in Phase 1 to compare Go vs TS extraction on the same input.
type ExtractionTrace struct {
	TopicID           string                       `json:"topic_id"`
	UserID            string                       `json:"user_id"`
	Content           string                       `json:"content"`
	Timestamp         time.Time                    `json:"timestamp"`
	LLMCalls          []LLMCallTrace               `json:"llm_calls"`
	ExtractedMemories ExtractedMemoriesTrace       `json:"extracted_memories"`
	PersistedCount    int                          `json:"persisted_count"`
	Errors            []string                     `json:"errors,omitempty"`
}

type LLMCallTrace struct {
	Layer    string `json:"layer"`
	Request  string `json:"request,omitempty"`
	Response string `json:"response"`
	Duration int    `json:"duration_ms"`
	Error    string `json:"error,omitempty"`
}

type ExtractedMemoriesTrace struct {
	Identities  []CreateIdentityInput  `json:"identities"`
	Activities  []CreateActivityInput  `json:"activities"`
	Contexts    []CreateContextInput   `json:"contexts"`
	Experiences []CreateExperienceInput `json:"experiences"`
	Preferences []CreatePreferenceInput `json:"preferences"`
}

// tracingLLM wraps an LLMClient and records all calls for debugging.
type tracingLLM struct {
	delegate LLMClient
	mu       sync.Mutex
	calls    []LLMCallTrace
}

func newTracingLLM(delegate LLMClient) *tracingLLM {
	return &tracingLLM{delegate: delegate}
}

func (t *tracingLLM) Chat(ctx context.Context, systemPrompt, userPrompt string) (string, error) {
	t.mu.Lock()
	startTime := time.Now()
	t.mu.Unlock()

	response, err := t.delegate.Chat(ctx, systemPrompt, userPrompt)

	t.mu.Lock()
	defer t.mu.Unlock()

	call := LLMCallTrace{
		Response: response,
		Duration: int(time.Since(startTime).Milliseconds()),
		Request:  userPrompt[:min(len(userPrompt), 100)], // truncate for brevity
	}

	// Infer layer from system prompt
	if bytes.Contains([]byte(systemPrompt), []byte("gatekeeper")) {
		call.Layer = "gatekeeper"
	} else if bytes.Contains([]byte(systemPrompt), []byte("identity")) {
		call.Layer = "identity"
	} else if bytes.Contains([]byte(systemPrompt), []byte("activity")) {
		call.Layer = "activity"
	} else if bytes.Contains([]byte(systemPrompt), []byte("context")) {
		call.Layer = "context"
	} else if bytes.Contains([]byte(systemPrompt), []byte("experience")) {
		call.Layer = "experience"
	} else if bytes.Contains([]byte(systemPrompt), []byte("preference")) {
		call.Layer = "preference"
	}

	if err != nil {
		call.Error = err.Error()
	}

	t.calls = append(t.calls, call)
	return response, err
}

// tracingStore wraps a Store and records all insertions for tracing.
type tracingStore struct {
	delegate Store
	mu       sync.Mutex
	extracted ExtractedMemoriesTrace
}

func newTracingStore(delegate Store) *tracingStore {
	return &tracingStore{delegate: delegate}
}

func (t *tracingStore) CreateIdentity(ctx context.Context, userID string, in CreateIdentityInput) (string, error) {
	id, err := t.delegate.CreateIdentity(ctx, userID, in)
	t.mu.Lock()
	defer t.mu.Unlock()
	t.extracted.Identities = append(t.extracted.Identities, in)
	return id, err
}

func (t *tracingStore) CreateActivity(ctx context.Context, userID string, in CreateActivityInput) (string, error) {
	id, err := t.delegate.CreateActivity(ctx, userID, in)
	t.mu.Lock()
	defer t.mu.Unlock()
	t.extracted.Activities = append(t.extracted.Activities, in)
	return id, err
}

func (t *tracingStore) CreateContext(ctx context.Context, userID string, in CreateContextInput) (string, error) {
	id, err := t.delegate.CreateContext(ctx, userID, in)
	t.mu.Lock()
	defer t.mu.Unlock()
	t.extracted.Contexts = append(t.extracted.Contexts, in)
	return id, err
}

func (t *tracingStore) CreateExperience(ctx context.Context, userID string, in CreateExperienceInput) (string, error) {
	id, err := t.delegate.CreateExperience(ctx, userID, in)
	t.mu.Lock()
	defer t.mu.Unlock()
	t.extracted.Experiences = append(t.extracted.Experiences, in)
	return id, err
}

func (t *tracingStore) CreatePreference(ctx context.Context, userID string, in CreatePreferenceInput) (string, error) {
	id, err := t.delegate.CreatePreference(ctx, userID, in)
	t.mu.Lock()
	defer t.mu.Unlock()
	t.extracted.Preferences = append(t.extracted.Preferences, in)
	return id, err
}

// OpenAILLM calls the OpenAI API (for real extraction in parity testing).
type OpenAILLM struct {
	apiKey  string
	baseURL string
	model   string
	client  *http.Client
}

func NewOpenAILLM(apiKey, model string) *OpenAILLM {
	return &OpenAILLM{
		apiKey:  apiKey,
		baseURL: "https://api.openai.com/v1",
		model:   model,
		client:  &http.Client{Timeout: 30 * time.Second},
	}
}

func (o *OpenAILLM) Chat(ctx context.Context, systemPrompt, userPrompt string) (string, error) {
	payload := map[string]any{
		"model":    o.model,
		"messages": []map[string]string{
			{"role": "system", "content": systemPrompt},
			{"role": "user", "content": userPrompt},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+o.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("openai: status %d: %s", resp.StatusCode, respBody)
	}

	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	if len(result.Choices) == 0 {
		return "", fmt.Errorf("openai: no choices in response")
	}

	return result.Choices[0].Message.Content, nil
}

// TestPhase1SingleTopicTrace runs extraction on a single topic using real OpenAI
// and outputs a JSON trace for manual comparison with TS.
//
// This is NOT a unit test — it requires:
// - OPENAI_API_KEY environment variable
// - Write permission to current directory (to save trace.json)
//
// Run with: OPENAI_API_KEY=sk-... go test -v -run TestPhase1SingleTopicTrace ./memoryingest
//
// Expected output:
// - phase1_extraction_trace.json (full trace including LLM calls and extracted memories)
func TestPhase1SingleTopicTrace(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping Phase 1 trace in short mode (requires OPENAI_API_KEY)")
	}

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set; skipping Phase 1 trace")
	}

	// Use a realistic topic for testing
	topic := MemoryIngestArgs{
		UserID:  "test-user-1",
		TopicID: "test-topic-1",
		Content: `
User: I'm working on optimizing our database queries. We're seeing 300ms latency on a join.
Assistant: Have you looked at the query plan? Try EXPLAIN ANALYZE.
User: Yeah, the join is sequential scanning a 500k row table without an index.
Assistant: Add a composite index on the join columns. What's your use case?
User: We're filtering by user_id and created_at. I made myself a coffee while it reindexed.
Assistant: That's a good combination for composites. Coffee's always good.
User: Finished. Latency dropped to 45ms. Thanks! This was my first database optimization project.
Assistant: Great! You've learned an important optimization technique today.
`,
	}

	// Setup tracing infrastructure
	llm := newTracingLLM(NewOpenAILLM(apiKey, "gpt-4o-mini"))
	store := newTracingStore(&fakeStore{})
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	// Run extraction
	worker := NewIngestWorker(Config{
		LLM:    llm,
		Store:  store,
		Logger: logger,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	err := worker.run(ctx, topic, logger)
	if err != nil {
		t.Fatalf("extraction failed: %v", err)
	}

	// Assemble trace
	trace := ExtractionTrace{
		TopicID:           topic.TopicID,
		UserID:            topic.UserID,
		Content:           topic.Content,
		Timestamp:         time.Now(),
		LLMCalls:          llm.calls,
		ExtractedMemories: store.extracted,
		PersistedCount:    len(store.extracted.Identities) + len(store.extracted.Activities) + len(store.extracted.Contexts) + len(store.extracted.Experiences) + len(store.extracted.Preferences),
	}

	// Write trace to file
	traceFile := "phase1_extraction_trace.json"
	data, err := json.MarshalIndent(trace, "", "  ")
	if err != nil {
		t.Fatalf("marshal trace: %v", err)
	}

	if err := os.WriteFile(traceFile, data, 0o644); err != nil {
		t.Fatalf("write trace: %v", err)
	}

	t.Logf("✅ Trace written to %s", traceFile)
	t.Logf("📊 Extracted: %d identities, %d activities, %d contexts, %d experiences, %d preferences",
		len(store.extracted.Identities),
		len(store.extracted.Activities),
		len(store.extracted.Contexts),
		len(store.extracted.Experiences),
		len(store.extracted.Preferences),
	)
	t.Logf("💬 LLM calls: %d", len(llm.calls))

	// Summary
	if len(llm.calls) == 0 {
		t.Error("expected LLM calls but got none")
	}
	if trace.PersistedCount == 0 {
		t.Logf("⚠️  No memories extracted (may be expected if gatekeeper rejected)")
	}
}

// TestPhase1MockBatch runs batch extraction on synthetic topics using mocked LLM.
// This validates the worker's ability to handle multiple topics without real API costs.
func TestPhase1MockBatch(t *testing.T) {
	topics := []struct {
		name    string
		approved bool
		llmReplies []string
	}{
		{
			name:    "personal_experience",
			approved: true,
			llmReplies: []string{
				`{"relevant": true, "reason": "personal experience"}`,
				`[{"description":"Marathon runner","type":"personal","role":"athlete"}]`,
				`[{"type":"running","narrative":"Completed my first marathon yesterday","status":"completed"}]`,
				`[{"title":"Marathon Training","description":"12-week training plan","type":"goal"}]`,
				`[{"situation":"Hit the wall at mile 20","action":"Slowed pace and walked a bit","keyLearning":"Proper pacing is crucial"}]`,
				`[{"suggestions":"Run early in the morning to avoid heat"}]`,
			},
		},
		{
			name:    "small_talk",
			approved: false,
			llmReplies: []string{
				`{"relevant": false, "reason": "only greetings"}`,
			},
		},
		{
			name:    "work_project",
			approved: true,
			llmReplies: []string{
				`{"relevant": true, "reason": "professional project"}`,
				`[{"description":"Tech lead at Acme Corp","type":"professional","role":"lead"}]`,
				`[{"type":"coding","narrative":"Deployed v2.1 to production","status":"completed"}]`,
				`[{"title":"v2.1 Release","description":"Shipped new search feature"}]`,
				`[{"situation":"Team disagreed on API design","action":"Ran a design review session","keyLearning":"Async discussion helps convergence"}]`,
				`[{"suggestions":"Use RFC process for major API changes","type":"process"}]`,
			},
		},
	}

	for _, tc := range topics {
		t.Run(tc.name, func(t *testing.T) {
			store := &fakeStore{}
			llm := &scriptedLLM{replies: tc.llmReplies}
			worker := NewIngestWorker(Config{Store: store, LLM: llm, Logger: slog.New(slog.NewTextHandler(io.Discard, nil))})

			err := worker.run(context.Background(), MemoryIngestArgs{
				UserID:  "user-1",
				TopicID: "topic-" + tc.name,
				Content: "Sample content for " + tc.name,
			}, slog.New(slog.NewTextHandler(io.Discard, nil)))
			if err != nil {
				t.Fatalf("run: %v", err)
			}

			if tc.approved {
				total := len(store.identities) + len(store.activities) + len(store.contexts) + len(store.experiences) + len(store.preferences)
				if total == 0 {
					t.Error("expected memories to be extracted")
				}
			} else {
				total := len(store.identities) + len(store.activities) + len(store.contexts) + len(store.experiences) + len(store.preferences)
				if total > 0 {
					t.Error("expected no memories when gatekeeper rejects")
				}
			}
		})
	}
}

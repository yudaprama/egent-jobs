package memoryingest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

// OpenAIChatClient is the production LLMClient. It posts
// system+user prompts to an OpenAI-compatible /chat/completions
// endpoint. Configure via env vars or constructor.
//
// Env vars:
//
//	MODEL_BASE_URL       — e.g. https://openrouter.ai/api/v1
//	MODEL_API_KEY        — bearer token; empty for local models
//	MEMORY_EXTRACT_MODEL — chat model id (default: gpt-4o-mini)
type OpenAIChatClient struct {
	baseURL string
	apiKey  string
	model   string
	client  *http.Client
}

// NewOpenAIChatClientFromEnv builds a client from MODEL_BASE_URL /
// MODEL_API_KEY / MEMORY_EXTRACT_MODEL. Returns nil when baseURL is
// unset (the worker can then run in tests with a scripted fake).
func NewOpenAIChatClientFromEnv() *OpenAIChatClient {
	baseURL := os.Getenv("MODEL_BASE_URL")
	if baseURL == "" {
		return nil
	}
	model := os.Getenv("MEMORY_EXTRACT_MODEL")
	if model == "" {
		model = "gpt-4o-mini"
	}
	return &OpenAIChatClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  os.Getenv("MODEL_API_KEY"),
		model:   model,
		client:  &http.Client{Timeout: 120 * 1_000_000_000}, // 120s
	}
}

// NewOpenAIChatClient builds a client with explicit values. Used by
// tests and by callers that need to override the env-var defaults.
func NewOpenAIChatClient(baseURL, apiKey, model string, httpClient *http.Client) *OpenAIChatClient {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 120 * 1_000_000_000}
	}
	return &OpenAIChatClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		model:   model,
		client:  httpClient,
	}
}

// Chat satisfies LLMClient.
func (c *OpenAIChatClient) Chat(ctx context.Context, systemPrompt, userPrompt string) (string, error) {
	if c == nil {
		return "", fmt.Errorf("memoryingest: nil OpenAI chat client")
	}
	body := map[string]any{
		"model": c.model,
		"messages": []map[string]string{
			{"role": "system", "content": systemPrompt},
			{"role": "user", "content": userPrompt},
		},
		"temperature": 0.2,
		"stream":      false,
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("memoryingest: marshal chat: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/chat/completions", bytes.NewReader(raw))
	if err != nil {
		return "", fmt.Errorf("memoryingest: build chat request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("memoryingest: chat call: %w", err)
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("memoryingest: read chat response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("memoryingest: chat returned %d: %s",
			resp.StatusCode, truncate(string(respBody), 500))
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return "", fmt.Errorf("memoryingest: decode chat response: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return "", fmt.Errorf("memoryingest: chat returned 0 choices")
	}
	return parsed.Choices[0].Message.Content, nil
}
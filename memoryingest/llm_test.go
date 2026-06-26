package memoryingest

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOpenAIChatClient_SendsAndParses(t *testing.T) {
	var gotBody map[string]any
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hello back"}}]}`))
	}))
	defer srv.Close()

	c := NewOpenAIChatClient(srv.URL, "k", "test-model", srv.Client())
	got, err := c.Chat(context.Background(), "system says hi", "user asks")
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if got != "hello back" {
		t.Errorf("got %q, want %q", got, "hello back")
	}
	if gotAuth != "Bearer k" {
		t.Errorf("auth = %q, want Bearer k", gotAuth)
	}
	if msgs, _ := gotBody["messages"].([]any); len(msgs) != 2 {
		t.Errorf("expected 2 messages, got %d", len(msgs))
	}
	if model, _ := gotBody["model"].(string); model != "test-model" {
		t.Errorf("model = %q", model)
	}
}

func TestOpenAIChatClient_Non2xxReturnsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		_, _ = w.Write([]byte(`{"error":"boom"}`))
	}))
	defer srv.Close()

	c := NewOpenAIChatClient(srv.URL, "", "m", srv.Client())
	_, err := c.Chat(context.Background(), "s", "u")
	if err == nil {
		t.Fatal("expected error on 500")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error should mention status, got %v", err)
	}
}

func TestOpenAIChatClient_EmptyChoicesReturnsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[]}`))
	}))
	defer srv.Close()

	c := NewOpenAIChatClient(srv.URL, "", "m", srv.Client())
	_, err := c.Chat(context.Background(), "s", "u")
	if err == nil {
		t.Fatal("expected error on empty choices")
	}
}

func TestNewOpenAIChatClientFromEnv_Disabled(t *testing.T) {
	t.Setenv("MODEL_BASE_URL", "")
	c := NewOpenAIChatClientFromEnv()
	if c != nil {
		t.Error("expected nil when MODEL_BASE_URL is empty")
	}
}
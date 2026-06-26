package fileingest

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHTTPFetcher_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("hello world"))
	}))
	defer srv.Close()

	f := &HTTPFetcher{}
	body, err := f.Fetch(context.Background(), srv.URL+"/file.txt")
	if err != nil {
		t.Fatalf("fetch failed: %v", err)
	}
	if string(body) != "hello world" {
		t.Fatalf("got %q", body)
	}
}

func TestHTTPFetcher_InlineSentinel(t *testing.T) {
	f := &HTTPFetcher{}
	_, err := f.Fetch(context.Background(), "internal://document/placeholder")
	if err == nil {
		t.Fatal("expected error for internal:// url")
	}
	if !strings.Contains(err.Error(), "inline document") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestHTTPFetcher_NoSuchKey(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()
	f := &HTTPFetcher{}
	_, err := f.Fetch(context.Background(), srv.URL+"/missing")
	if !IsStorageNoSuchKey(err) {
		t.Fatalf("expected NoSuchKey sentinel, got %v", err)
	}
}

func TestHTTPFetcher_Empty(t *testing.T) {
	f := &HTTPFetcher{}
	_, err := f.Fetch(context.Background(), "")
	if err == nil {
		t.Fatal("expected error for empty url")
	}
}

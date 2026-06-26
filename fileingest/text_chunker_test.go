package fileingest

import (
	"context"
	"strings"
	"testing"
)

func TestTextChunker_Basic(t *testing.T) {
	c := &TextChunker{MaxChunkBytes: 200}
	content := []byte("para one.\n\npara two.\n\npara three.\n")
	rows, err := c.Chunk(context.Background(), ChunkInput{
		Filename: "a.md",
		FileType: "text/markdown",
		Content:  content,
	})
	if err != nil {
		t.Fatalf("chunk failed: %v", err)
	}
	if len(rows) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(rows))
	}
	for i, r := range rows {
		if r.Text == "" {
			t.Fatalf("chunk %d is empty", i)
		}
		if r.Type != "DocumentChunk" {
			t.Fatalf("chunk %d has wrong type: %s", i, r.Type)
		}
	}
}

func TestTextChunker_LongLineSplit(t *testing.T) {
	c := &TextChunker{MaxChunkBytes: 50}
	longLine := strings.Repeat("x", 120)
	rows, err := c.Chunk(context.Background(), ChunkInput{
		Filename: "long.txt",
		FileType: "text/plain",
		Content:  []byte(longLine),
	})
	if err != nil {
		t.Fatalf("chunk failed: %v", err)
	}
	if len(rows) < 2 {
		t.Fatalf("expected at least 2 chunks from long line, got %d", len(rows))
	}
}

func TestTextChunker_NotUTF8(t *testing.T) {
	c := &TextChunker{}
	_, err := c.Chunk(context.Background(), ChunkInput{
		Filename: "binary.pdf",
		FileType: "application/pdf",
		Content:  []byte{0xff, 0xfe, 0xfd},
	})
	if err == nil {
		t.Fatal("expected UTF-8 error")
	}
}

func TestTextChunker_Empty(t *testing.T) {
	c := &TextChunker{}
	rows, err := c.Chunk(context.Background(), ChunkInput{Content: nil})
	if err != nil {
		t.Fatalf("empty should not error: %v", err)
	}
	if len(rows) != 0 {
		t.Fatalf("expected 0 chunks, got %d", len(rows))
	}
}

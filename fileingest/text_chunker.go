package fileingest

import (
	"bufio"
	"context"
	"fmt"
	"strings"
	"unicode/utf8"
)

// TextChunker is a pragmatic default Chunker that splits UTF-8 text on
// double newlines (paragraph) and on a per-paragraph cap. It is NOT a
// replacement for the production TS ContentChunk pipeline (which uses
// unstructured.io to handle PDFs, DOCX, etc.) — its job is to give the
// River worker something that works on plain-text / markdown uploads out
// of the box so the integration is testable end-to-end without standing
// up the full TS chunker in Go.
//
// Operators that need PDF / DOCX / rich parsing should swap in a Chunker
// implementation that shells out to an external partitioner (or a future
// Go-native port of the ContentChunk rules).
type TextChunker struct {
	// MaxChunkBytes caps each chunk. 0 → 1500 (matches LobeHub's default).
	MaxChunkBytes int
}

func (c *TextChunker) Chunk(_ context.Context, in ChunkInput) ([]ChunkRow, error) {
	if len(in.Content) == 0 {
		return nil, nil
	}
	cap := c.MaxChunkBytes
	if cap <= 0 {
		cap = 1500
	}
	// Quick-and-dirty: treat the file as UTF-8 text and split on blank
	// lines. Binary files (PDF, DOCX) will surface as a single chunk
	// containing the raw bytes — which is wrong but at least doesn't crash.
	// The default fetcher's max size + the NoSuchKey guard keep this from
	// running on genuinely missing content.
	if !utf8.Valid(in.Content) {
		return nil, fmt.Errorf("textchunker: file is not valid UTF-8 (filetype=%s); install a real chunker", in.FileType)
	}
	scanner := bufio.NewScanner(strings.NewReader(string(in.Content)))
	scanner.Buffer(make([]byte, 0, 64*1024), cap*2)

	var (
		rows     []ChunkRow
		buf      strings.Builder
		idx      int
		flushBuf = func() {
			if buf.Len() == 0 {
				return
			}
			rows = append(rows, ChunkRow{
				Text:     strings.TrimSpace(buf.String()),
				Type:     "DocumentChunk",
				Metadata: map[string]any{"source": "textchunker", "index": idx},
			})
			idx++
			buf.Reset()
		}
	)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			flushBuf()
			continue
		}
		// Split long lines on the cap.
		for len(line) > cap {
			flushBuf()
			rows = append(rows, ChunkRow{Text: line[:cap], Type: "DocumentChunk", Metadata: map[string]any{"source": "textchunker", "index": idx}})
			idx++
			line = line[cap:]
			buf.WriteString(line + "\n")
			break
		}
		buf.WriteString(line + "\n")
		if buf.Len() >= cap {
			flushBuf()
		}
	}
	flushBuf()
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("textchunker: scan: %w", err)
	}
	return rows, nil
}

package fileingest

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// HTTPFetcher is the default Fetcher. It treats fileService URLs as HTTPS
// endpoints and downloads the raw bytes — which is the equivalent of
// fileService.getFileByteArray(file.url) in async/file.ts.
//
// In production, file URLs point at AList (https://assets.getkawai.com/.../xxx)
// or Supabase Storage. For the inline `internal://document/placeholder` marker
// the TS handler returns early; this Go worker also bails on that prefix to
// preserve the legacy behavior.
type HTTPFetcher struct {
	Client  *http.Client
	MaxSize int64 // bytes; 0 means no limit. Defaults to 100 MB.
}

func (f *HTTPFetcher) Fetch(ctx context.Context, url string) ([]byte, error) {
	if url == "" {
		return nil, fmt.Errorf("fileingest: empty url")
	}
	if strings.HasPrefix(url, "internal://") {
		// Mirrors the TS guard: inline documents are searched via BM25 and
		// do not get chunked. Return a sentinel error so the worker can
		// surface a clear message to the operator.
		return nil, fmt.Errorf("fileingest: inline document (url=%q) is skipped", url)
	}
	client := f.Client
	if client == nil {
		client = &http.Client{Timeout: 5 * time.Minute}
	}
	maxSize := f.MaxSize
	if maxSize == 0 {
		maxSize = 100 << 20 // 100 MB
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("fileingest: build request for %s: %w", url, err)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fileingest: fetch %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// NoSuchKey detection matches the TS handler's branch: the
		// caller (worker) records the task as Error and does NOT retry.
		if resp.StatusCode == http.StatusNotFound {
			return nil, errStorageNoSuchKey
		}
		return nil, fmt.Errorf("fileingest: fetch %s returned %d", url, resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxSize+1))
	if err != nil {
		return nil, fmt.Errorf("fileingest: read %s: %w", url, err)
	}
	if int64(len(body)) > maxSize {
		return nil, fmt.Errorf("fileingest: file too large (>%d bytes)", maxSize)
	}
	return body, nil
}

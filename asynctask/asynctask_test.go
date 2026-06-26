package asynctask

import "testing"

func TestConstantsMatchTS(t *testing.T) {
	// These string values are observed by the BFF via Tier 1 pREST queries
	// and via direct async_tasks.status reads. Drift here = silent BFF
	// desync. Pin them in this test.
	cases := []struct {
		got, want string
	}{
		{StatusPending, "pending"},
		{StatusProcessing, "processing"},
		{StatusSuccess, "success"},
		{StatusError, "error"},
		{ErrorTypeTimeout, "Timeout"},
		{ErrorTypeEmbeddingError, "EmbeddingError"},
		{ErrorTypeContentPolicy, "ContentPolicyError"},
		{QueueFileIngest, "file_ingest"},
		{QueueMediaGen, "media_gen"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("const drift: got %q want %q", c.got, c.want)
		}
	}
}

func TestFirstNonEmpty(t *testing.T) {
	if firstNonEmpty("", "x", "y") != "x" {
		t.Fatal("expected first non-empty")
	}
	if firstNonEmpty("", "") != "" {
		t.Fatal("expected empty when all empty")
	}
	if firstNonEmpty("a") != "a" {
		t.Fatal("expected a")
	}
}

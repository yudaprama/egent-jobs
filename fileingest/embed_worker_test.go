package fileingest

import (
	"context"
	"testing"

	"egent-jobs/embeddings"
)

// fakeEmbedder is a deterministic Embedder used in unit tests. It produces a
// fixed vector per chunk so assertions don't depend on a real HTTP call.
type fakeEmbedder struct {
	dim     int
	model   string
	vectors map[string][]float32
	err     error
	calls   int
}

func (f *fakeEmbedder) Model() string { return f.model }

func (f *fakeEmbedder) EmbedBatch(_ context.Context, inputs []embeddings.Input) ([]embeddings.Result, error) {
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	out := make([]embeddings.Result, len(inputs))
	for i, in := range inputs {
		v, ok := f.vectors[in.ChunkID]
		if !ok {
			v = make([]float32, f.dim)
			for j := range v {
				v[j] = float32(len(in.ChunkID) + j)
			}
		}
		out[i] = embeddings.Result{ChunkID: in.ChunkID, Vector: v}
	}
	return out, nil
}

func TestChunkBy(t *testing.T) {
	cases := []struct {
		name string
		in   []int
		size int
		want [][]int
	}{
		{"empty", nil, 3, [][]int{}},
		{"exact", []int{1, 2, 3}, 3, [][]int{{1, 2, 3}}},
		{"overflow", []int{1, 2, 3, 4, 5}, 3, [][]int{{1, 2, 3}, {4, 5}}},
		{"size_zero_clamps_to_1", []int{1, 2}, 0, [][]int{{1}, {2}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := chunkBy(tc.in, tc.size)
			if len(got) != len(tc.want) {
				t.Fatalf("len mismatch: got %d want %d", len(got), len(tc.want))
			}
			for i := range got {
				if len(got[i]) != len(tc.want[i]) {
					t.Fatalf("batch %d len mismatch: got %d want %d", i, len(got[i]), len(tc.want[i]))
				}
				for j := range got[i] {
					if got[i][j] != tc.want[i][j] {
						t.Fatalf("batch %d index %d: got %d want %d", i, j, got[i][j], tc.want[i][j])
					}
				}
			}
		})
	}
}

func TestVectorToPG(t *testing.T) {
	got := vectorToPG([]float32{1.5, -2.25, 0})
	want := "[1.5,-2.25,0]"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestVectorToPG_Dimension(t *testing.T) {
	// Sanity check that the produced literal matches the vector length —
	// critical because public.embeddings is vector(1024).
	v := make([]float32, 1024)
	for i := range v {
		v[i] = float32(i) * 0.001
	}
	out := vectorToPG(v)
	// Count commas — should be len-1.
	commas := 0
	for _, c := range out {
		if c == ',' {
			commas++
		}
	}
	if commas != 1023 {
		t.Fatalf("expected 1023 commas for 1024-dim vector, got %d", commas)
	}
	if len(out) < 2 || out[0] != '[' || out[len(out)-1] != ']' {
		t.Fatalf("malformed literal: %q", out[:min(20, len(out))])
	}
}

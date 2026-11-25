//go:build integration

package integration

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// Match upstream's requireSimilarEmbedding tolerance in embed_test.go.
// Owner decision (2026-09-27), rationale and prior drift measurements:
// docs/parallel-embedding-validation.md, "Similarity tolerance".
const parallelEmbeddingThreshold = 0.999

var parallelEmbeddingPrompts = []string{
	"A red fox crosses the frozen river at dawn.",
	"fn scoreLexicalMatch(query: []const u8) f32",
	"SELECT document_id FROM embeddings ORDER BY distance LIMIT 10;",
	"Parallel slots must preserve vector ordering and identity.",
	"The quick brown fox jumps over the lazy dog.",
	"Embedding corruption can look valid while poisoning retrieval.",
}

// Validate before using the suite's cosine helper: malformed vectors must not
// pass through NaN comparisons, empty dot products, or float32 accumulation.
func parallelEmbeddingCosine(a, b []float32) (float64, error) {
	if len(a) == 0 || len(a) != len(b) {
		return 0, fmt.Errorf("empty or mismatched dimensions")
	}
	x, y := make([]float64, len(a)), make([]float64, len(b))
	for i := range a {
		x[i], y[i] = float64(a[i]), float64(b[i])
		if math.IsNaN(x[i]) || math.IsNaN(y[i]) || math.IsInf(x[i], 0) || math.IsInf(y[i], 0) {
			return 0, fmt.Errorf("non-finite embedding")
		}
	}
	if magnitude(x) == 0 || magnitude(y) == 0 {
		return 0, fmt.Errorf("zero-norm embedding")
	}
	return cosineSimilarity(x, y), nil
}

func TestEmbeddingParallelOracle(t *testing.T) {
	for _, bad := range [][]float32{nil, {1}, {0, 0}, {float32(math.NaN()), 1}, {float32(math.Inf(1)), 1}} {
		if _, err := parallelEmbeddingCosine([]float32{1, 2}, bad); err == nil {
			t.Errorf("accepted malformed vector %v", bad)
		}
	}
	if score, err := parallelEmbeddingCosine([]float32{1, 2}, []float32{3, 6}); err != nil || score < parallelEmbeddingThreshold {
		t.Fatalf("scaled vector: cosine=%g err=%v", score, err)
	}
	if score, err := parallelEmbeddingCosine([]float32{1, 0}, []float32{0, 1}); err != nil || score >= parallelEmbeddingThreshold {
		t.Fatalf("swapped identity: cosine=%g err=%v", score, err)
	}
}

type parallelEmbeddingFixture struct {
	name, path, hash    string
	context, dimensions int
}

func embeddingFileHash(t *testing.T, path string) string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func embeddingEvidence(t *testing.T, path string, value any) {
	t.Helper()
	b, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Error(err)
		return
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Error(err)
	}
}

// This gate needs immutable local fixtures; ordinary integration runs retain
// their existing model selection. Explicit gate modes fail on missing fixtures.
func TestEmbeddingParallel(t *testing.T) {
	mode := os.Getenv("OLLAMA_TEST_EMBED_MODE")
	if mode == "" {
		t.Skip("set OLLAMA_TEST_EMBED_MODE=release or cpu-regression for pinned local fixtures")
	}
	if mode != "release" && mode != "cpu-regression" {
		t.Fatalf("unknown embedding mode %q", mode)
	}
	fixtures := []parallelEmbeddingFixture{{"all-minilm", os.Getenv("OLLAMA_TEST_MINILM"), "797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662", 256, 384}}
	devices := []string{"cpu"}
	if mode == "release" {
		fixtures = append(fixtures, parallelEmbeddingFixture{"jina", os.Getenv("OLLAMA_TEST_JINA"), "4b257dde63951a2365418841135552acb1daa3e3beab6fed1e53cdeb09061522", 8192, 1536})
		devices = append(devices, "cuda")
	}
	for _, fixture := range fixtures {
		if got := embeddingFileHash(t, fixture.path); got != fixture.hash {
			t.Fatalf("%s: expected SHA256 %s, got %s", fixture.name, fixture.hash, got)
		}
	}
	output := os.Getenv("OLLAMA_TEST_EMBED_OUTPUT")
	if output == "" {
		output = filepath.Join(t.TempDir(), "evidence")
	}
	if err := os.Mkdir(output, 0o755); err != nil {
		t.Fatal(err)
	}
	summary := map[string]any{"mode": mode, "threshold": parallelEmbeddingThreshold}
	var cases []map[string]any
	t.Cleanup(func() {
		if len(cases) != len(fixtures)*len(devices)*5 {
			t.Error("incomplete embedding matrix; a filtered run is not a release pass")
		}
		summary["cases"] = cases
		summary["passed"] = !t.Failed()
		embeddingEvidence(t, filepath.Join(output, "summary.json"), summary)
	})
	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			baselines := map[string][][]float32{}
			for _, device := range devices {
				t.Run(device, func(t *testing.T) {
					for slots := 1; slots <= 5; slots++ {
						ran := false
						passed := t.Run(strconv.Itoa(slots), func(t *testing.T) {
							ran = true
							vectors := runParallelEmbeddingCase(t, fixture, device, slots, baselines[device], output)
							if slots == 1 {
								baselines[device] = vectors
							}
						})
						if ran {
							cases = append(cases, map[string]any{"model": fixture.name, "device": device, "slots": slots, "passed": passed})
						}
					}
				})
			}
			if len(baselines["cpu"]) > 0 && len(baselines["cuda"]) > 0 {
				minimum := 1.0
				for i, vector := range baselines["cpu"] {
					score, err := parallelEmbeddingCosine(vector, baselines["cuda"][i])
					if err != nil {
						t.Fatal(err)
					}
					minimum = min(minimum, score)
				}
				// Different backends are diagnostic; each must preserve its own
				// single-slot vectors without relaxing the threshold.
				summary[fixture.name+"_cpu_cuda_cosine"] = minimum
				t.Logf("CPU/CUDA one-slot cosine: %.12f", minimum)
			}
		})
	}
}

func runParallelEmbeddingCase(t *testing.T, fixture parallelEmbeddingFixture, device string, slots int, baseline [][]float32, output string) [][]float32 {
	t.Helper()
	out := filepath.Join(output, fmt.Sprintf("%s-%s-%d", fixture.name, device, slots))
	if err := os.Mkdir(out, 0o755); err != nil {
		t.Fatal(err)
	}
	result := map[string]any{"model_sha256": fixture.hash, "device": device, "slots": slots, "context": fixture.context, "dimensions": fixture.dimensions, "threshold": parallelEmbeddingThreshold, "prompts": parallelEmbeddingPrompts}
	t.Cleanup(func() {
		result["passed"] = !t.Failed()
		embeddingEvidence(t, filepath.Join(out, "result.json"), result)
	})
	// Preserve test controls, but prevent ambient server settings from changing
	// the requested device, context, slots or numerical configuration.
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		if (strings.HasPrefix(key, "OLLAMA_") && !strings.HasPrefix(key, "OLLAMA_TEST_")) || strings.HasPrefix(key, "GGML_") || strings.HasPrefix(key, "LLAMA_") {
			t.Setenv(key, "")
		}
	}
	state := t.TempDir()
	models := filepath.Join(state, "models")
	if err := os.MkdirAll(filepath.Join(models, "blobs"), 0o755); err != nil {
		t.Fatal(err)
	}
	// Ollama touches blob timestamps, so copy rather than link immutable input.
	src, err := os.Open(fixture.path)
	if err != nil {
		t.Fatal(err)
	}
	dst, err := os.Create(filepath.Join(models, "blobs", "sha256-"+fixture.hash))
	if err != nil {
		src.Close()
		t.Fatal(err)
	}
	_, copyErr := io.Copy(dst, src)
	src.Close()
	closeErr := dst.Close()
	if copyErr != nil {
		t.Fatal(copyErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	for key, value := range map[string]string{
		"OLLAMA_TEST_HOME": state, "OLLAMA_TEST_EXISTING": "", "OLLAMA_MODELS": models,
		"OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_NUM_PARALLEL": strconv.Itoa(slots),
		"OLLAMA_CONTEXT_LENGTH": strconv.Itoa(fixture.context), "OLLAMA_NO_CLOUD": "1",
		"OLLAMA_KEEP_ALIVE": "10m", "OLLAMA_LOAD_TIMEOUT": "30s", "OLLAMA_DEBUG": "1",
	} {
		t.Setenv(key, value)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer func() {
		cleanup()
		log := serverLog.String() // Read only after the process writer has stopped.
		if err := os.WriteFile(filepath.Join(out, "server.log"), []byte(log), 0o644); err != nil {
			t.Error(err)
		}
		commands := regexp.MustCompile(`msg="starting llama-server" cmd="([^"]+)"`).FindAllStringSubmatch(log, -1)
		observed := false
		for _, command := range commands {
			observed = observed || strings.Contains(command[1], fmt.Sprintf(" -np %d ", slots))
		}
		result["runner_commands"] = commands
		if !observed {
			t.Errorf("requested slot count %d not observed in runner log", slots)
		}
		if device == "cuda" {
			offloads := regexp.MustCompile(`offloaded (\d+)/(\d+) layers to GPU`).FindAllStringSubmatch(log, -1)
			if len(offloads) == 0 || !strings.Contains(log, "library=CUDA") {
				t.Error("CUDA offload not observed")
			}
			for _, offload := range offloads {
				if offload[1] != offload[2] || offload[1] == "0" {
					t.Error("expected full CUDA offload")
				}
			}
		}
	}()
	result["binary"] = serverCmd.Path
	result["binary_sha256"] = embeddingFileHash(t, serverCmd.Path)
	version, err := client.Version(ctx)
	if err != nil {
		t.Fatal(err)
	}
	result["version"] = version
	no := false
	err = client.Create(ctx, &api.CreateRequest{Model: "probe", Files: map[string]string{"model.gguf": "sha256:" + fixture.hash}, Template: "{{ .Prompt }}", Parameters: map[string]any{"num_ctx": fixture.context}, Stream: &no}, func(api.ProgressResponse) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	info, err := client.Show(ctx, &api.ShowRequest{Model: "probe"})
	if err != nil {
		t.Fatal(err)
	}
	result["model_info"] = info
	options := map[string]any{"num_ctx": fixture.context, "num_gpu": 0, "num_thread": 8}
	if device == "cuda" {
		options["num_gpu"] = 99
	}
	result["options"] = options
	var responses []map[string]any
	defer func() { embeddingEvidence(t, filepath.Join(out, "responses.json"), responses) }()
	embed := func(indices []int) [][]float32 {
		input := make([]string, len(indices))
		for i, index := range indices {
			input[i] = parallelEmbeddingPrompts[index]
		}
		resp, err := client.Embed(ctx, &api.EmbedRequest{Model: "probe", Input: input, Truncate: &no, Options: options})
		if err != nil {
			t.Fatal(err)
		}
		responses = append(responses, map[string]any{"indices": indices, "response": resp})
		if len(resp.Embeddings) != len(input) {
			t.Fatal("wrong embedding count")
		}
		for _, vector := range resp.Embeddings {
			if len(vector) != fixture.dimensions {
				t.Fatalf("expected %d dimensions, got %d", fixture.dimensions, len(vector))
			}
			if _, err := parallelEmbeddingCosine(vector, vector); err != nil {
				t.Fatal(err)
			}
		}
		return resp.Embeddings
	}
	var checks []map[string]any
	defer func() { result["checks"] = checks }()
	check := func(name string, indices []int, vectors, reference [][]float32) {
		minimum := 1.0
		for i, index := range indices {
			score, err := parallelEmbeddingCosine(reference[index], vectors[i])
			if err != nil {
				t.Fatal(err)
			}
			minimum = min(minimum, score)
		}
		checks = append(checks, map[string]any{"name": name, "minimum_cosine": minimum, "passed": minimum >= parallelEmbeddingThreshold})
		if minimum < parallelEmbeddingThreshold {
			t.Errorf("%s: cosine %.12f < %.4f", name, minimum, parallelEmbeddingThreshold)
		}
	}
	serial := func() [][]float32 {
		vectors := make([][]float32, len(parallelEmbeddingPrompts))
		for i := range vectors {
			vectors[i] = embed([]int{i})[0]
		}
		return vectors
	}
	reference := serial()
	result["serial_vectors"] = reference
	distinct := false
	for _, vector := range reference[1:] {
		score, _ := parallelEmbeddingCosine(reference[0], vector)
		distinct = distinct || score < parallelEmbeddingThreshold
	}
	if !distinct {
		t.Fatal("reference does not distinguish different input texts")
	}
	order := []int{0, 1, 2, 3, 4, 5}
	for repeat := range 2 {
		check(fmt.Sprintf("serial_repeat_%d", repeat), order, serial(), reference)
	}
	for _, batch := range []struct {
		name    string
		indices []int
	}{{"batch", order}, {"repeat", order}, {"reverse", []int{5, 4, 3, 2, 1, 0}}, {"duplicate_neighbors", []int{3, 0, 3, 5, 1, 2, 4, 0}}} {
		check(batch.name, batch.indices, embed(batch.indices), reference)
	}
	if slots > 1 {
		if len(baseline) != len(reference) {
			t.Fatal("one-slot reference unavailable")
		}
		for i, response := range responses {
			check(fmt.Sprintf("one_slot_response_%d", i), response["indices"].([]int), response["response"].(*api.EmbedResponse).Embeddings, baseline)
		}
	}
	loaded, err := client.ListRunning(ctx)
	if err != nil {
		t.Fatal(err)
	}
	result["loaded"] = loaded
	if len(loaded.Models) != 1 {
		t.Fatal("expected one loaded model")
	}
	if vram := loaded.Models[0].SizeVRAM; (device == "cpu" && vram != 0) || (device == "cuda" && vram <= 0) {
		t.Errorf("wrong device: VRAM=%d", vram)
	}
	return reference
}

package llm

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

// Captured b11081 formats; these fixtures are deliberately independent of the
// matcher's expression. Keep both the must-suppress and must-preserve corpus.
var routineLogFixtures = []string{
	"srv        update:    - prompt 0x602d5bfc3e60:       2 tokens, checkpoints:  0,     0.055 MiB\n",
	"srv         alloc:  - prompt is already in the cache, skipping\n",
	"srv   prompt_save:  - saving prompt with length 7, total state size = 0.192 MiB (draft: 0.000 MiB)\n",
	"srv  update_slots: all slots are idle\n",
	"slot process_sing: id  0 | task -1 | saving idle slot to prompt cache\n",
	"slot launch_slot_: id  0 | task 82 | processing task, is_child = 0\n",
	"slot   operator(): id  0 | task 82 | cached n_tokens = 0, memory_seq_rm [0, end)\n",
	"slot get_availabl: id  0 | task 82 |  - skipping, is_processing = 1\n",
	"slot get_availabl: id  0 | task -1 |  - checking sim = 0.000 (0/7) > 0.100\n",
	"slot get_availabl: id  0 | task -1 |  - skipping, slot is empty\n",
	"slot   operator(): id  0 | task 82 | new prompt, n_ctx_slot = 8192, n_keep = 0, task.n_tokens = 7\n",
	"slot      release: id  0 | task 82 | stop processing: n_tokens = 7, truncated = 0\n",
	"srv        update:  - cache state: 12 prompts, 0.055 MiB (limits: 8192.000 MiB, 24576 tokens, 120 est)\n",
	"slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1\n",
	"slot get_availabl: id  0 | task -1 | selected slot by LCP similarity, f_sim_best = 0.500 (> 0.100 thold), f_keep = 0.500\n",
}

var accountingLogFixtures = []string{
	"using device CUDA0 (NVIDIA GPU) (0000:01:00.0) - 12000 MiB free\n",
	"load_tensors:        CUDA0 model buffer size =  1000.00 MiB\n",
	"llama_kv_cache:      CUDA0 KV buffer size =  2000.00 MiB\n",
	"sched_reserve:      CUDA0 compute buffer size =   300.00 MiB\n",
	"load_tensors:   CPU_Mapped model buffer size =   100.00 MiB\n",
	"llm_load_tensors: offloaded 33/33 layers to GPU\n",
}

func writeLog(t *testing.T, w io.Writer, s string) {
	t.Helper()
	if n, err := w.Write([]byte(s)); err != nil || n != len(s) {
		t.Fatalf("Write = %d, %v; want %d, nil", n, err, len(s))
	}
}

func TestRunnerLogStartupAndRestart(t *testing.T) {
	var out bytes.Buffer
	w := newRunnerLogWriter(&out)
	startup := strings.Join(routineLogFixtures, "") + "loading..."
	writeLog(t, w, startup)
	if out.String() != startup {
		t.Fatal("startup output was buffered or filtered")
	}
	w.ready()
	writeLog(t, w, routineLogFixtures[0]) // completes the startup fragment
	if out.String() != startup+routineLogFixtures[0] {
		t.Fatal("startup fragment was reclassified")
	}
	for range 1000 {
		writeLog(t, w, routineLogFixtures[0])
	}
	if err := w.flush(); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	w.start()
	writeLog(t, w, startup)
	if out.String() != startup {
		t.Fatal("retry did not restore verbatim startup")
	}
}

func TestRunnerLogFragmentationAndSpecificity(t *testing.T) {
	keep := []string{
		"srv llama_server: model loaded\n",
		"srv        update: warning: unusual cache state\n",
		"slot      release: CUDA error: out of memory\n",
		"srv        update: unfamiliar future format\n",
		"plain unknown log line\n",
	}
	// Mutate every known routine body with a diagnostic suffix and prefix.
	// The matcher must not swallow diagnostics merely because a prefix matches.
	for _, line := range routineLogFixtures {
		keep = append(keep, strings.TrimSuffix(line, "\n")+" WARNING: changed\n", "error: "+line)
	}
	input := strings.Repeat(strings.Join(routineLogFixtures, ""), 30) + strings.Join(keep, "") + "unterminated tail"
	for _, chunk := range []int{1, 2, 3, 7, 31, 128, 4096, len(input)} {
		t.Run(fmt.Sprint(chunk), func(t *testing.T) {
			var out bytes.Buffer
			w := newRunnerLogWriter(&out)
			w.now = func() time.Time { return time.Unix(100, 0) }
			w.ready()
			for offset := 0; offset < len(input); offset += chunk {
				writeLog(t, w, input[offset:min(offset+chunk, len(input))])
			}
			if err := w.flush(); err != nil {
				t.Fatal(err)
			}
			for _, line := range keep {
				if !strings.Contains(out.String(), line) {
					t.Errorf("lost diagnostic %q", line)
				}
			}
			if !strings.HasSuffix(out.String(), "unterminated tail") {
				t.Fatal("lost tail")
			}
			if got := strings.Count(out.String(), "\n"); got != 8+len(keep)+1 {
				t.Fatalf("forwarded %d lines, want %d", got, 8+len(keep)+1)
			}
			if !strings.Contains(out.String(), "suppressed 442 repetitive runtime lines") {
				t.Fatal("missing accurate suppression counter")
			}
		})
	}
}

func TestRunnerLogWindowAndUnknownFlood(t *testing.T) {
	var out bytes.Buffer
	now := time.Unix(100, 0)
	w := newRunnerLogWriter(&out)
	w.now = func() time.Time { return now }
	w.ready()
	for range 1000 {
		writeLog(t, w, routineLogFixtures[0])
	}
	for range 250 {
		writeLog(t, w, "unknown warning\n")
	}
	now = now.Add(30 * time.Second)
	writeLog(t, w, routineLogFixtures[0])
	if got := strings.Count(out.String(), routineLogFixtures[0]); got != 9 {
		t.Fatalf("samples = %d", got)
	}
	if got := strings.Count(out.String(), "unknown warning\n"); got != 250 {
		t.Fatalf("warnings = %d", got)
	}
	if !strings.Contains(out.String(), "suppressed 992 repetitive runtime lines") {
		t.Fatal("missing window counter")
	}
	if err := w.flush(); err != nil {
		t.Fatal(err)
	}
	if strings.Count(out.String(), "suppressed") != 1 {
		t.Fatal("duplicate summary")
	}
}

func TestRunnerLogOverlongFailOpen(t *testing.T) {
	var out bytes.Buffer
	w := newRunnerLogWriter(&out)
	w.ready()
	input := strings.Repeat("x", 100_000) + routineLogFixtures[0]
	for i := 0; i < len(input); i += 127 {
		writeLog(t, w, input[i:min(i+127, len(input))])
		if len(w.pending) > 8192 {
			t.Fatal("unbounded pending line")
		}
	}
	if err := w.flush(); err != nil {
		t.Fatal(err)
	}
	if out.String() != input {
		t.Fatal("overlong line or its suffix was modified")
	}
}

type failingLogWriter struct {
	n   int
	err error
}

func (w failingLogWriter) Write([]byte) (int, error) { return w.n, w.err }

func TestRunnerLogOutputErrors(t *testing.T) {
	for _, running := range []bool{false, true} {
		for _, err := range []error{nil, errors.New("output failed")} {
			w := newRunnerLogWriter(failingLogWriter{err: err})
			if running {
				w.ready()
			}
			_, got := w.Write([]byte("warning\n"))
			want := err
			if want == nil {
				want = io.ErrShortWrite
			}
			if !errors.Is(got, want) {
				t.Fatalf("error = %v, want %v", got, want)
			}
			if n, got := w.Write([]byte("later\n")); n != 0 || !errors.Is(got, want) {
				t.Fatal("output error was lost")
			}
			if !errors.Is(w.flush(), want) {
				t.Fatal("flush lost output error")
			}
		}
	}
}

type logParserSnapshot struct {
	Total, GPU, Model, CPUMapped, Layers, TotalLayers uint64
	Overflow                                          int
	ByDevice, Free                                    map[string]uint64
	LastError                                         string
}

func logSnapshot(r *llamaServerRunner, s *StatusWriter) logParserSnapshot {
	return logParserSnapshot{r.memTotal, r.memGPU, r.memModelFileBacked, r.memCPUMappedModel, r.gpuLayers, r.totalLayers, r.gpuLayerOverflow, r.vramByDevice, r.systemFreeAtLoad, s.LastError()}
}
func logPipeline(out io.Writer) (*memoryParsingWriter, *StatusWriter) {
	s := NewStatusWriter(out)
	r := &llamaServerRunner{systemFreeAtLoad: make(map[string]uint64), vramByDevice: make(map[string]uint64)}
	return &memoryParsingWriter{runner: r, inner: s}, s
}

func TestRunnerLogRawParserParity(t *testing.T) {
	input := strings.Join(accountingLogFixtures, "") + strings.Repeat(strings.Join(routineLogFixtures, ""), 20) +
		"srv        update: warning: sentinel\nCUDA error: failed to allocate sentinel\n"
	// Existing parsers are chunk-sensitive. Compare identical raw chunks, not
	// an invented claim that filtering repairs those pre-existing parsers.
	for _, chunk := range []int{1, 2, 7, 37, 128, 4096, len(input)} {
		t.Run(fmt.Sprint(chunk), func(t *testing.T) {
			var out bytes.Buffer
			candidate, candidateStatus, filter := newRunnerOutput(&out)
			candidate.runner = &llamaServerRunner{systemFreeAtLoad: make(map[string]uint64), vramByDevice: make(map[string]uint64)}
			filter.ready()
			baseline, baselineStatus := logPipeline(io.Discard)
			for offset := 0; offset < len(input); offset += chunk {
				part := input[offset:min(offset+chunk, len(input))]
				writeLog(t, candidate, part)
				writeLog(t, baseline, part)
			}
			if err := filter.flush(); err != nil {
				t.Fatal(err)
			}
			got, want := logSnapshot(candidate.runner, candidateStatus), logSnapshot(baseline.runner, baselineStatus)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("parser state changed: %+v != %+v", got, want)
			}
			if !strings.Contains(out.String(), "CUDA error: failed to allocate sentinel\n") {
				t.Fatal("lost error output")
			}
			if chunk == len(input) {
				if got.Total != 3400<<20 || got.GPU != 3300<<20 || got.Model != 1100<<20 || got.CPUMapped != 100<<20 || got.Layers != 33 || got.TotalLayers != 33 || got.Free["CUDA0"] != 12000<<20 || got.ByDevice["CUDA0"] != 3300<<20 || got.LastError != "CUDA error: failed to allocate sentinel" {
					t.Fatalf("accounting/error control failed: %+v", got)
				}
			}
		})
	}
}

func TestRunnerLogConcurrentReadyAndWrites(t *testing.T) {
	var out bytes.Buffer
	w := newRunnerLogWriter(&out)
	w.now = func() time.Time { return time.Unix(100, 0) }
	w.ready()
	var workers sync.WaitGroup
	for range 8 {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for range 100 {
				w.ready() // repeated health checks cannot reset the sample budget
				writeLog(t, w, routineLogFixtures[0])
			}
		}()
	}
	workers.Wait()
	if err := w.flush(); err != nil {
		t.Fatal(err)
	}
	if strings.Count(out.String(), routineLogFixtures[0]) != 8 || !strings.Contains(out.String(), "suppressed 792 repetitive runtime lines") {
		t.Fatal("concurrent writes or ready notifications reset the budget")
	}
}

func TestRunnerLogSuppressionStillReachesRawConsumer(t *testing.T) {
	var out bytes.Buffer
	input, _, filter := newRunnerOutput(&out)
	input.runner = &llamaServerRunner{}
	input.runner.startLoadTracking(time.Now())
	filter.ready()
	for range 100 {
		// The memory parser records raw activity even for routine bytes that
		// will be suppressed. Putting the filter before it breaks this control.
		input.runner.loadActivity.Store(1)
		writeLog(t, input, routineLogFixtures[0])
		if input.runner.loadActivity.Load() == 1 {
			t.Fatal("suppression prevented the raw consumer from seeing input")
		}
	}
	if strings.Count(out.String(), routineLogFixtures[0]) != 8 {
		t.Fatal("control did not exercise suppressed writes")
	}
}

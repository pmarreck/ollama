package llm

import (
	"bytes"
	"fmt"
	"io"
	"regexp"
	"sync"
	"time"
)

const (
	runnerLogWindow  = 30 * time.Second
	runnerLogSamples = 8
	runnerLogMaxLine = 8 * 1024
)

// runnerLogWriter filters only the final forwarded copy, AFTER memoryParsingWriter
// and StatusWriter have consumed the original bytes. Startup is always verbatim.
// Unknown formats fail open; adding a format requires evidence from llama.cpp.
// See docs/runner-logging.md for provenance and qualification limits.
type runnerLogWriter struct {
	mu         sync.Mutex
	out        io.Writer
	now        func() time.Time
	running    bool
	passLine   bool // startup fragment or an overlong line; never classify its suffix
	pending    []byte
	window     time.Time
	sampled    int
	suppressed uint64
	err        error
}

func newRunnerLogWriter(out io.Writer) *runnerLogWriter {
	return &runnerLogWriter{out: out, now: time.Now}
}

// Keep the production composition shared with the raw-parser and journal
// regression controls. The caller assigns memory.runner before starting it.
func newRunnerOutput(out io.Writer) (*memoryParsingWriter, *StatusWriter, *runnerLogWriter) {
	forward := newRunnerLogWriter(out)
	status := NewStatusWriter(forward)
	return &memoryParsingWriter{inner: status}, status, forward
}

// start is called before each launch, including a retry after startup failure.
func (w *runnerLogWriter) start() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.running = false
	w.passLine = false
	w.pending = w.pending[:0]
	w.sampled, w.suppressed = 0, 0
}

func (w *runnerLogWriter) ready() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.running {
		w.running = true
		w.window = w.now()
	}
}

func (w *runnerLogWriter) emit(b []byte) error {
	n, err := w.out.Write(b)
	if err == nil && n != len(b) {
		err = io.ErrShortWrite
	}
	w.err = err
	return err
}

func (w *runnerLogWriter) summary() error {
	if w.suppressed == 0 {
		return nil
	}
	err := w.emit([]byte(fmt.Sprintf("ollama runner logs: suppressed %d repetitive runtime lines (sample limit %d per %s)\n", w.suppressed, runnerLogSamples, runnerLogWindow)))
	w.suppressed = 0
	return err
}

func (w *runnerLogWriter) line(b []byte) error {
	now := w.now()
	if now.Sub(w.window) >= runnerLogWindow {
		if err := w.summary(); err != nil {
			return err
		}
		w.window, w.sampled = now, 0
	}
	if routineRunnerLine.Match(bytes.TrimSuffix(b, []byte{'\n'})) {
		if w.sampled >= runnerLogSamples {
			w.suppressed++
			return nil
		}
		w.sampled++
	}
	return w.emit(b)
}

func (w *runnerLogWriter) Write(b []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.err != nil {
		return 0, w.err
	}
	if len(b) == 0 {
		return 0, nil
	}
	if !w.running {
		w.passLine = b[len(b)-1] != '\n'
		n, err := w.out.Write(b)
		if err == nil && n != len(b) {
			err = io.ErrShortWrite
		}
		w.err = err
		return n, err
	}

	// n counts input accepted (including intentionally suppressed bytes), not
	// output bytes. On an output error, stop accepting input and retain the error.
	n := 0
	for len(b) > 0 {
		end := bytes.IndexByte(b, '\n')
		complete := end >= 0
		if complete {
			end++
		} else {
			end = len(b)
		}
		part := b[:end]
		if !w.passLine && len(w.pending)+len(part) > runnerLogMaxLine {
			if err := w.emit(w.pending); err != nil {
				return n, err
			}
			w.pending = w.pending[:0]
			w.passLine = true
		}
		if w.passLine {
			if err := w.emit(part); err != nil {
				return n, err
			}
			w.passLine = !complete
		} else {
			w.pending = append(w.pending, part...)
			if complete {
				if err := w.line(w.pending); err != nil {
					return n, err
				}
				w.pending = w.pending[:0]
			}
		}
		n += len(part)
		b = b[end:]
	}
	return n, nil
}

// flush runs after cmd.Wait has drained both output streams. Unterminated tails
// are never suppressed. There is no timer goroutine and no unbounded queue.
func (w *runnerLogWriter) flush() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.err != nil {
		return w.err
	}
	// Put the summary before an unterminated tail so it cannot become part of it.
	// An overlong tail was already forwarded; finish that record first.
	if w.passLine && w.suppressed > 0 {
		if err := w.emit([]byte{'\n'}); err != nil {
			return err
		}
		w.passLine = false
	}
	if err := w.summary(); err != nil {
		return err
	}
	if len(w.pending) > 0 {
		if err := w.emit(w.pending); err != nil {
			return err
		}
		w.pending = w.pending[:0]
	}
	return nil
}

// Exact b11081 cache/slot bodies observed in the incident journal. Numeric
// fields vary; arbitrary trailing text, warning/error bodies and new formats do
// not match. Do not replace this with a blanket srv/slot prefix filter.
var routineRunnerLine = regexp.MustCompile(`^(?:` +
	`srv +update: + - prompt 0x[0-9a-f]+: +[0-9]+ tokens, checkpoints: +[0-9]+, +[0-9]+\.[0-9]+ MiB|` +
	`srv +update: + - cache state: [0-9]+ prompts, [0-9]+\.[0-9]+ MiB \(limits: [0-9]+\.[0-9]+ MiB, [0-9]+ tokens, [0-9]+ est\)|` +
	`srv +alloc: + - prompt is already in the cache, skipping|` +
	`srv +prompt_save: + - saving prompt with length [0-9]+, total state size = [0-9]+\.[0-9]+ MiB \(draft: [0-9]+\.[0-9]+ MiB\)|` +
	`srv +update_slots: all slots are idle|` +
	`slot +(?:` +
	`process_sing: id +[0-9]+ \| task -?[0-9]+ \| saving idle slot to prompt cache|` +
	`launch_slot_: id +[0-9]+ \| task [0-9]+ \| processing task, is_child = [01]|` +
	`operator\(\): id +[0-9]+ \| task -?[0-9]+ \| (?:cached n_tokens = [0-9]+, memory_seq_rm \[[0-9]+, end\)|new prompt, n_ctx_slot = [0-9]+, n_keep = [0-9]+, task.n_tokens = [0-9]+)|` +
	`get_availabl: id +[0-9]+ \| task -?[0-9]+ \|  - (?:skipping, is_processing = [01]|checking sim = [0-9]+\.[0-9]+ \([0-9]+/[0-9]+\) > [0-9]+\.[0-9]+|skipping, slot is empty)|` +
	`get_availabl: id +[0-9]+ \| task -?[0-9]+ \| selected slot by (?:LRU, t_last = -?[0-9]+|LCP similarity, f_sim_best = [0-9]+\.[0-9]+ \(> [0-9]+\.[0-9]+ thold\), f_keep = [0-9]+\.[0-9]+)|` +
	`release: id +[0-9]+ \| task -?[0-9]+ \| stop processing: n_tokens = [0-9]+, truncated = [01])` +
	`)$`)

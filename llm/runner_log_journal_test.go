//go:build linux

package llm

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// This test runs ONLY in the disposable NixOS VM. The same compiled Go pipeline
// supplies the baseline and candidate; journald, not an in-memory approximation,
// independently decides which records survive the unchanged unit rate limit.
func TestRunnerJournal(t *testing.T) {
	if os.Getenv("OLLAMA_LOG_JOURNAL_VM") != "1" {
		t.Skip("requires disposable journald VM")
	}
	if marker, err := os.ReadFile("/etc/ollama-log-test-vm"); err != nil || string(marker) != "disposable\n" {
		t.Fatal("refusing journal test outside marked disposable VM")
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	var baseline logParserSnapshot
	for _, mode := range []string{"baseline", "candidate"} {
		t.Run(mode, func(t *testing.T) {
			unit := "ollama-log-test-" + mode
			report := filepath.Join(t.TempDir(), "parser.json")
			cmd := exec.Command("systemd-run", "--wait", "--unit="+unit,
				"--property=Type=oneshot", "--property=StandardOutput=journal", "--property=StandardError=journal",
				"--property=LogRateLimitIntervalSec=30s", "--property=LogRateLimitBurst=200", "--property=RuntimeMaxSec=30s",
				"--setenv=OLLAMA_LOG_JOURNAL_EMITTER="+mode, "--setenv=OLLAMA_LOG_JOURNAL_REPORT="+report,
				exe, "-test.run=^TestRunnerJournalEmitter$", "-test.v")
			if output, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("emitter: %v\n%s", err, output)
			}
			if output, err := exec.Command("journalctl", "--sync").CombinedOutput(); err != nil {
				t.Fatalf("sync: %v\n%s", err, output)
			}
			data, err := os.ReadFile(report)
			if err != nil {
				t.Fatal(err)
			}
			var snapshot logParserSnapshot
			if err := json.Unmarshal(data, &snapshot); err != nil {
				t.Fatal(err)
			}
			if mode == "baseline" {
				baseline = snapshot
			} else if !reflect.DeepEqual(snapshot, baseline) {
				t.Fatalf("raw parser state changed: %+v != %+v", snapshot, baseline)
			}
			if snapshot.Total != 3400<<20 || snapshot.GPU != 3300<<20 || snapshot.Layers != 33 || snapshot.Free["CUDA0"] != 12000<<20 || snapshot.LastError != "CUDA error: sentinel-after" {
				t.Fatalf("independent accounting/error expectations failed: %+v", snapshot)
			}
			journal, err := exec.Command("journalctl", "--unit="+unit, "--output=json", "--no-pager").Output()
			if err != nil {
				t.Fatal(err)
			}
			var messages []string
			scanner := bufio.NewScanner(bytes.NewReader(journal))
			for scanner.Scan() {
				var row struct {
					Message string `json:"MESSAGE"`
				}
				if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
					t.Fatal(err)
				}
				messages = append(messages, row.Message)
			}
			if err := scanner.Err(); err != nil {
				t.Fatal(err)
			}
			joined := strings.Join(messages, "\n")
			if mode == "baseline" {
				if len(messages) < 200 || strings.Contains(joined, "sentinel-after") {
					t.Fatalf("negative control did not hit real journald limiting: %d records", len(messages))
				}
			} else {
				for _, sentinel := range []string{"sentinel-startup", "warning: sentinel-during", "CUDA error: sentinel-after", "unknown-format sentinel-after", "suppressed 29992 repetitive runtime lines"} {
					if !strings.Contains(joined, sentinel) {
						t.Errorf("journal lost %q", sentinel)
					}
				}
				if len(messages) > 40 {
					t.Errorf("candidate journal still floods: %d records", len(messages))
				}
			}
			t.Logf("%s: %d journal records; parser state %+v", mode, len(messages), snapshot)
		})
	}
}

func TestRunnerJournalEmitter(t *testing.T) {
	mode := os.Getenv("OLLAMA_LOG_JOURNAL_EMITTER")
	if mode == "" {
		t.Skip("VM subprocess only")
	}
	if mode != "baseline" && mode != "candidate" {
		t.Fatal("invalid emitter mode")
	}
	pipeline, status, filter := newRunnerOutput(os.Stdout)
	pipeline.runner = &llamaServerRunner{systemFreeAtLoad: make(map[string]uint64), vramByDevice: make(map[string]uint64)}
	if mode == "baseline" {
		pipeline, status = logPipeline(os.Stdout)
	}
	writeLog(t, pipeline, "sentinel-startup\n")
	for _, line := range accountingLogFixtures {
		writeLog(t, pipeline, line)
	}
	filter.ready()
	for i := range 30000 {
		if i == 15000 {
			writeLog(t, pipeline, "srv        update: warning: sentinel-during\n")
		}
		// Fragment and batch real cache-dump messages across Write boundaries.
		line := fmt.Sprintf("srv        update:    - prompt 0x602d5bfc3e60:       %d tokens, checkpoints:  0,     0.055 MiB\n", i%16)
		writeLog(t, pipeline, line[:17])
		writeLog(t, pipeline, line[17:])
	}
	writeLog(t, pipeline, "CUDA error: sentinel-after\nunknown-format sentinel-after\n")
	if mode == "candidate" {
		if err := filter.flush(); err != nil {
			t.Fatal(err)
		}
	}
	data, err := json.Marshal(logSnapshot(pipeline.runner, status))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(os.Getenv("OLLAMA_LOG_JOURNAL_REPORT"), data, 0600); err != nil {
		t.Fatal(err)
	}
}

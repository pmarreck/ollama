# Runner log forwarding

Ollama must retain llama-server verbosity 4 for startup memory, device and
layer-offload accounting. Reducing that verbosity indiscriminately can change
scheduler decisions. Instead, the forwarding chain is:

```text
raw subprocess bytes -> memoryParsingWriter -> StatusWriter -> runnerLogWriter -> stderr
```

Both existing parsers receive the original writes before the final copy is
filtered. Startup output passes through immediately, including fragments without
newlines. Filtering starts only after the health endpoint reports ready, and
returns to startup behavior before any process retry.

Once ready, the writer samples at most eight recognized routine lines per
30-second window, per runner. It emits one suppressed-line count at the next
window's first complete line, or at process exit. There is no timer or per-line
queue. This bounds routine output, not all service logging: startup, unknown
formats, warnings and errors deliberately remain unrestricted. Several runners
share the service's journal budget. An unknown new flood needs investigation,
not a broader prefix filter.

Only complete, anchored cache and slot formats are eligible. The fixtures came
from a retained b11081 incident journal; the cache/slot calls can be checked in
[llama.cpp b11081 server-context.cpp](https://github.com/ggml-org/llama.cpp/blob/b11081/tools/server/server-context.cpp).
Arbitrary suffixes or prefixes fail open. In particular, `srv` and `slot` alone
are not suppression criteria. Fragmented and batched writes are reassembled
only at this final forwarding boundary. Pending content is capped at 8 KiB;
overlong lines pass through, as do unterminated tails at process exit.

The memory and status parsers retain their pre-existing chunk-sensitive
behavior. Differential tests send identical chunks through the original and
filtered chains; they do not claim that this patch repairs parser fragmentation.
Whole-record controls separately verify memory totals, device free memory,
offloaded layers, and captured errors against known values.

## Why a pin update was not selected

On November 25, 2026, upstream Ollama `784dd544291825e8f5f064672dc2bd7f4d9345ed`
used llama.cpp b12651. It still passed verbosity 4 and forwarded runner output
unchanged; llama.cpp still classified the repetitive cache/slot records at trace
level 4. Upstream change `adc4aae9d` disabled the prompt cache for decision
models, which does not establish a fix for the Jina embedding path. Updating
these pins alone was therefore not a demonstrated repair.

In the retained incident sample (14:00–17:45 EST), 12,923 of 14,162 records
started with `srv` or `slot`; 10,171 were `srv update` records. Gin accounted for
146 retained records. The largest aligned 30-second Gin bucket contained 14.
These are censored counts: journald had already discarded records, so they do
not establish the actual arrival rate or a peak bound. Gin uses a separate
path and is unchanged. A journal suppression count denotes lost log records,
not failed inference requests or an explanation for a socket failure.

## Qualification

The existing Go suite supplies the tests. Build and execute the sandboxed checks
through the shared resource budget:

```sh
build-budget --project ollama -- nix build \
  .#checks.x86_64-linux.runner-logs \
  .#checks.x86_64-linux.runner-journal --no-link -L
```

`runner-logs` runs the full `llm` Go suite with Go's race detector
inside a Nix build sandbox. It covers startup/retry, fragmented and batched
writes, known-format mutations, unknown warnings, errors, window accounting,
unterminated/overlong lines, output failures and raw-parser parity.

`runner-journal` boots a disposable NixOS VM. NixOS's standard driver only boots
the guest and invokes the compiled Go test. Replay and all assertions live in
`llm/runner_log_journal_test.go`. The original and filtered paths each emit
30,000 routine records into separate units with the same 200 messages / 30 seconds
base limit used by the affected service. The negative control must demonstrate
real journal loss; the candidate must retain startup, middle and final diagnostic
sentinels, match raw parser state, and leave fewer than 40 journal records. The
VM has no production service or model access. This is journal qualification,
not an inference-load or CUDA performance test.

On November 25, 2026, all 96 top-level `llm` tests passed with the race detector
on x86_64 Linux. The disposable journal negative control lost the final error
sentinel; the candidate retained all required sentinels and reported 29,992
suppressed routine records out of 30,000. The baseline and candidate agreed on
3,400 MiB total memory, 3,300 MiB GPU memory, 33 offloaded layers, 12,000 MiB
device free memory, and the final captured CUDA error. The journal test's Go
output records the retained record counts on each run.

The native package and embedding regression remain separate required checks.
Host installer/service tests must not run on the live system: an installer test
can restart the daemon even when `OLLAMA_NO_START` is set. Existing CI does not
by itself certify GPU execution, Metal/JetPack, UI, or production behavior.

Committing and pushing this work does not authorize production activation,
service restart or changes to service/host journal policy. A production cutover
requires separate owner approval and subsequent observation of the real workload.

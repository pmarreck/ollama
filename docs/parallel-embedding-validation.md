# Parallel embedding validation

Parallel embedding support is opt-in and model-dependent. A successful build
does not prove that multiple request slots preserve the vectors or their order.

## Pooled graph reservation fix

llama.cpp b11081 rounds reserve tokens up to a multiple of the number of slots,
but leaves the output count unchanged. With mean pooling, the resulting matrix
dimensions disagree when the batch size is not divisible by the slot count.
The fork backports [llama.cpp PR #27186](https://github.com/ggml-org/llama.cpp/pull/27186),
at `91321ed05416f347101fabc7e12c72f2609680ae`, to make the pooled output count
match the rounded token count during graph reservation. Remove the patch when
the pinned llama.cpp includes the fix.

On September 27, 2026, the same all-minilm GGUF reproduced the following on an
AMD Threadripper 3990X / NVIDIA RTX 3080 Ti, with CUDA 12.9 and an sm_86 build:

| Package | Device | Slots | Result |
| --- | --- | --- | --- |
| 0.34.4-pmarreck.1 | CPU | 1, 2, 4 | Pass |
| 0.34.4-pmarreck.1 | CPU | 3, 5 | `ggml_can_mul_mat` assertion during load |
| 0.34.4-pmarreck.1 | CUDA | 1 | Pass |
| 0.34.4-pmarreck.1 | CUDA | 3, 5 | Same assertion |
| 0.34.4-pmarreck.2, patched | CPU | 1–5 | Pass; minimum cosine 0.9999997569 |
| 0.34.4-pmarreck.2, patched | CUDA | 1–5 | Pass; minimum cosine 0.9999974781 |

The comparisons include repeated singletons, concurrent batches, reversed
ordering, duplicate inputs with different neighbors, and a separate one-slot
server as the reference. The one-slot CPU/CUDA comparison was 0.9999983527.
These measurements establish the fix for this fixture and configuration;
they do not certify every embedding model.

## Reproducible gate

The gate pins these exact files:

| Model | GGUF SHA-256 | Context | Dimensions |
| --- | --- | --- | --- |
| all-minilm | `797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662` | 256 | 384 |
| Installed Jina code embeddings 1.5B Q8 | `4b257dde63951a2365418841135552acb1daa3e3beab6fed1e53cdeb09061522` | 8192 | 1536 |

The Jina file contains `qwen2.pooling_type=3`. Downloading a fresh publisher
GGUF and adding pooling metadata does not establish that it is the same file.
Supply a readable copy of the installed blob; the gate rejects a different hash.

```sh
package="$(nix build .#default --no-link --print-out-paths)"
fixture="$(nix build .#embedding-test-model --no-link --print-out-paths)"
export OLLAMA_TEST_BINARY="$package/bin/ollama"
export OLLAMA_TEST_MINILM="$fixture"
export OLLAMA_TEST_JINA=/path/to/jina-installed.gguf
export OLLAMA_TEST_EMBED_MODE=release
export OLLAMA_TEST_EMBED_OUTPUT=/tmp/ollama-embedding-release-evidence
ulimit -c 0
nix develop -c go test -tags=integration,release -count=1 -timeout=30m \
  -run '^TestEmbeddingParallel' -v ./integration
```

Use a new output directory for each run. The all-minilm fixture uses a
fixed-output Nix fetch. These cases extend the existing Go integration suite,
using its API client and server lifecycle helpers. Tests start temporary
localhost servers and private model directories.
The installed service is not reconfigured or restarted.

A full release pass requires both models on CPU and CUDA at slot counts 1–5.
Each backend must preserve its own one-slot reference at cosine >= 0.999.
The gate rejects malformed, empty, zero-norm, non-finite, incorrectly sized,
and indistinguishable reference vectors. It checks the observed runner slot
count and requires full CUDA offload when CUDA is requested. CPU/GPU arithmetic
differences are reported separately for diagnosis. Each case uses eight CPU
threads.

The Go test exit status and `summary.json` give the verdict. Each case retains
`result.json`, raw `responses.json`, and `server.log`, recording model and executable hashes,
package version, options, runner arguments, vectors and comparisons. Any failed
case, missing required model, or filtered/incomplete matrix fails the command.
Do not release a package for parallel embedding use on the strength of a
build-only or partial gate result.

## Similarity tolerance

On September 27, 2026, Peter chose to use **0.999** for now, matching
[upstream's `requireSimilarEmbedding`](https://github.com/ollama/ollama/blob/8a5d4431480895bd3dba9936e4e4159b16643629/integration/embed_test.go#L63-L74).
Both the Go gate and the legacy `test_embedding_parallel` script now default to that
threshold. Upstream uses it for different embedding test cases; it is not an
upstream guarantee of serial/parallel equivalence for every model.

The previous **0.9999** cutoff was a fork-specific choice introduced in commit
`080ef099`, then carried into the Go gate. The observed Jina CUDA minimum of
about **0.99977** fails that stricter cutoff but satisfies **0.999**. This is an
explicit acceptance-policy change, not a correction to CUDA arithmetic. No
degradation of search quality has been demonstrated by these vector tests.

Retain the measured similarities in the evidence even when they pass. The
model hashes, dimensions, finite/nonzero-vector checks, ordering checks, actual
slot counts, and full CUDA-offload requirements continue to apply. Revisit the
tolerance if retrieval-quality measurements or upstream guidance justify it.

## CI scope and Jina status

`checks.<system>.embedding` runs the Go numerical-oracle rejection tests and the
all-minilm CPU matrix in the Nix sandbox. Mechatron runs this check in addition
to the package build. It catches the original assertion without requiring a
GPU or access to the installed Jina file. A green CPU check is explicitly not
a full CUDA/Jina release verdict.

The September 27 Go tests used the exact installed Jina file identified above,
superseding the September 24 experiments with a reconstructed GGUF. Each run
used context 8192, eight CPU threads, effective batch/microbatch 2048, and a
separate one-slot reference on the same backend.

The following verdicts record the original runs at **0.9999**; their evidence
has not been rewritten to use the new threshold.

| Package | Jina CPU, slots 1–5 at 0.9999 | Jina CUDA, slots 1–5 at 0.9999 |
| --- | --- | --- |
| Installed 0.32.13-pmarreck.1 | Pass | Matrix fails |
| Unpatched 0.34.4-pmarreck.1 | Pass | Matrix fails |
| Patched 0.34.4-pmarreck.2 | Pass | Matrix fails |

The patched package's repeated singleton requests agree, but concurrent batches
at slots 3–5 fall below 0.9999 (minimum observed 0.9997704304). Slot 2 passes
in that run but fails in the unpatched run, so a two-slot limit is not a
validated workaround. CPU comparisons pass at every tested slot count.
The patched one-slot CPU/CUDA comparison is 0.9997914366; that cross-backend
measurement is diagnostic, separate from the within-backend gate failures.

At the former 0.9999 cutoff, disabling flash attention on the patched package
also fails at slots 2–5 (minimum 0.9998276541).
The same failure class in the installed package establishes that it predates
this update; these tests do not identify the responsible CUDA kernel.

The all-minilm graph-reservation fix is verified. The recorded Jina CUDA
differences are within the newly accepted 0.999 tolerance; changing that
tolerance does not eliminate the differences. Fresh gate runs record their
own threshold and verdict in `summary.json`.

A fresh full Go gate run on September 27 at **0.999**, against
`0.34.4-pmarreck.2`, passed **all 20 cases**: both pinned models, CPU and CUDA,
and slots 1–5. The lowest within-backend cosine was **0.9997704304**. This
confirms acceptance under the documented tolerance, with the measured drift
still present.

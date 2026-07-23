# Fork it, flake it, fix it

- [x] Fork `ollama/ollama` to `pmarreck/ollama` and configure the local checkout. (2026-07-22 20:57 EDT)
  - Curiosity poke: preserve upstream `main` and keep the fork easy to rebase.
- [x] Add a failing scheduler test proving embedding models honor `OLLAMA_NUM_PARALLEL`. (2026-07-22 21:07 EDT)
  - Curiosity poke: models with known unsafe architectures must remain serial.
- [x] Remove the unconditional embedding-only `parallel=1` clamp. (2026-07-22 21:08 EDT)
  - Curiosity poke: default behavior must remain one slot when the variable is unset.
- [x] Add a Nix flake for reproducible development, tests, and CUDA-enabled packaging. (2026-07-22 21:16 EDT)
  - Curiosity poke: avoid depending on mutable host CUDA state during evaluation.
- [x] Document the fork's behavioral difference and the serial-versus-parallel correctness oracle. (2026-07-22 21:09 EDT)
  - Curiosity poke: silent vector corruption matters more than crashes or benchmark speed.
- [x] Run the reproducible native build and live three-slot CUDA oracle. (2026-07-23 00:07 EDT)
  - Curiosity poke: the runner command must prove `-np 3`, while singleton-versus-batch vectors must remain equivalent rather than merely well-shaped.

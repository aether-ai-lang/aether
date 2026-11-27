# Changelog

All notable changes to Aether. Formats follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] — Production Release

### Added
- **`sys.core`** — `read_file("path")`, `write_file("path", content)`, and
  `load_json("file", key: "field")` compiled to native C99 file I/O (zero Python).
- **HTTPS in `llm.core`** — `ask llm(...)` now speaks `https://` endpoints:
  Windows Schannel via runtime-loaded WinHTTP, POSIX via runtime-loaded libcurl;
  deterministic `AETHER_LLM_MOCK=1` offline mode preserved for CI.
- **Multi-OS CI** — GitHub Actions running the full unittest suite on
  ubuntu-latest, macos-latest, and windows-latest on every push/PR to main.
- **Official landing page** — `docs/index.html`, ready for GitHub Pages.
- `AETHER_API_KEY` bearer auth for the native HTTP server (see SECURITY.md).

## [0.8.0] — Packages, Planning & the OpenCL Tier

### Added
- **`aether init <name>`** — project scaffolding: `aether.toml` manifest, `src/main.ae`,
  `README.md`, `.gitignore`, `.aether_packages/`.
- **`aether install <source>`** — installs reusable `.ae` modules from git URLs,
  local directories, or single files into `.aether_packages/<name>/`; imported via
  `use pkg.<name>` with compile-time function merging and clear diagnostics for
  uninstalled packages.
- **`plan("goal") -> max_steps(N)`** — declarative multi-step agent planning through
  the LLM pipeline (deterministic offline mock), returning a numbered plan.
- **OpenCL dispatch tier** — `--target=opencl` emits a host/kernel plan that loads
  the vendor ICD at *runtime* (no OpenCL SDK needed to compile); probing requires a
  registered ICD, and missing runtimes get honest fallback telemetry.
- Auto-dispatch ladder extended: **CUDA → OpenCL → C99 → Python**.

## [0.7.0] — LLM Orchestration from Machine Code

### Added
- **`llm.core` / `ask llm("prompt") -> model("name")`** — OpenAI-compatible chat
  completions compiled to raw C99 socket code (Winsock2/POSIX), with
  `AETHER_LLM_URL`/`AETHER_LLM_KEY`/`AETHER_LLM_MOCK=1` environment contract and
  auto-mock when credentials are absent.
- **Tensor math** — `min(t)`, `max(t)`, `argmax(t)` (flat index → `int`), and
  element-wise scaling `t * 0.5`, valid inside larger expressions.
- **String building** — `+` concatenation for strings, `str(value)` conversion.
- **CUDA training kernels** — `--target=cuda --emit-only` emits real
  forward/backward SGD kernels (fused GEMM+bias+ReLU, softmax-CE, transposed GEMMs
  with atomic accumulation, SGD-momentum, gradient clipping) parameterized by the
  program's own intents.
- `examples/llm_agent_workflow.ae` — CSV → train → LLM summary → JSON.

## [0.6.0] — Standalone Binary, Persistent Memory & Reductions

### Added
- **Standalone binary** — `python build_standalone.py` → `dist/aether.exe`
  (19 MB, PyInstaller onefile) embedding the compiler, Python/NumPy backend, and
  the language spec; verified with Python removed from `PATH`.
- **Persistent agent memory** — `remember("key", value)` / `recall("key", default)`
  backed by `<program>.memory.json` (plain JSON, identical bytes from both
  backends), enabling agents that improve across separate executions.
- **Tensor reductions** — `mean(t)` / `sum(t)` compiled to OpenMP-parallel C loops.
- Frozen-binary support (`aetherc/bootstrap.py`) so the MCP server and bench work
  inside the standalone build.

## [0.5.0] — GPU Dispatch, data.core & Guarded Loops

### Added
- **Hardware probing and dispatch** — `auto_hardware` probes NVIDIA/AMD/Intel GPUs
  (`wmic`, `nvidia-smi -L`) and `nvcc`; ladder: CUDA → C99 (AVX-512/OpenBLAS) → Python,
  with explicit fallback notes in telemetry and JSON.
- **`data.core`** — `load_csv("file.csv", target: "col")` (numeric features, string-label
  interning, runtime shape discovery) and `save_json(value, "path")`, compiled to
  native C99 file I/O.
- **`while cond max_iters(N):`** — conditional loops with an infinite-loop guard
  that emits `while_guard` events, plus the built-in `accuracy` binding after `train`.
- Defender exclusions + high-priority child execution for stable benchmarking
  (dev-environment only).

## [0.4.0] — Turing-Complete Control Flow & agent.core

### Added
- **Scalars & arithmetic** — `let`, `+ - * / %` with C-style precedence, unary minus;
  `/` is true division on both backends.
- **`repeat N times:`** — counted loops gated on `agent.core`.
- **`if` / `else:`** — comparisons `== != > < >= <=`.
- **`log`** — structured output events.
- Type-stable `let` bindings and full compile-time type rules.
- `examples/autonomous_loop.ae` — self-retraining pipeline (accuracy 0.844 → 0.896 → 0.941
  across loop iterations).

## [0.3.0] — Native C99 HTTP Server & MCP

### Added
- **`web.core` / `serve`** — 4-line HTTP/JSON model servers compiled to standalone
  C99 (Winsock2/POSIX, thread-per-connection, ~0.1 ms per request), with a
  semantically identical Python stdlib fallback.
- **`load_model("model.ae.bin")`** — AEB1 model loading (shared binary format with
  the Python backend).
- **`aether mcp`** — Model Context Protocol server (JSON-RPC 2.0 over stdio) with
  six tools: check, self-heal (`aether_fix`), compile, run, spec, bench.
- `aether_fix` applies combined self-healing patches and re-validates.

## [0.2.0] — Initial Release: Compiler, C99 Backend & Self-Healing

### Added
- Full compiler pipeline: indentation-aware lexer → recursive-descent parser →
  type checker with intent contracts → code generator.
- **Two backends**: native C99 (`--target=c`, OpenMP + OpenBLAS, adaptive BLAS
  thread pool) and Python/NumPy — byte-identical AEB1 model exports.
- **Self-healing diagnostics** — stable codes (`AE0210`, `AE0300`, …), hints, and
  combined `fix_patch` unified diffs applicable with `git apply`.
- **`aether bench`** — Aether vs hand-written Python/NumPy/PyTorch: 96% fewer
  tokens, 1.68× faster than NumPy.
- Training intents with SGD + momentum, gradient clipping, and self-healing
  guards; streaming inference; `aether.toml`-free zero-config CLI.
- 19 end-to-end tests.

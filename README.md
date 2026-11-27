<div align="center">

# ⬡ Aether (`.ae`)

### *The First Programming Language Built for AI Agents, Not Human Typists.*

You declare **what** the AI workload is. Aether decides **how** to run it — and ships
as a **single standalone binary** that compiles your intent to multi-threaded C99
machine code: training jobs, streaming inference, native HTTP servers, **LLM
orchestration from raw sockets**, GPU dispatch, real CSV→model→JSON pipelines, and
**persistent agent memory**. Your agent drives everything over **MCP** — and Aether
itself no longer needs Python installed.

![license](https://img.shields.io/badge/license-MIT-blue)
![standalone](https://img.shields.io/badge/standalone-aether.exe%20%28zero%20Python%29-critical)
![backend](https://img.shields.io/badge/backend-C99%20%2B%20OpenMP%20%2B%20Winsock2-orange)
![gpu](https://img.shields.io/badge/GPU-CUDA%20%3E%20OpenCL%20%3E%20C99%20dispatch-blueviolet)
![packages](https://img.shields.io/badge/packages-aether%20init%20%2F%20install-teal)
![llm](https://img.shields.io/badge/llm.core-native%20C99%20client%20%2B%20mock%20mode-ff69b4)
![mcp](https://img.shields.io/badge/MCP-server%20%28stdio%20JSON--RPC%29-9cf)
![tests](https://img.shields.io/badge/tests-40%2F40-brightgreen)
![tokens](https://img.shields.io/badge/tokens-96%25%20saved-success)

**Load real CSV data. Train in a convergence loop. Export model + JSON report.
Nine lines, zero Python at runtime.**

```aether
use ai.core
use data.core
use agent.core

task RealDataPipeline:
    dataset = load_csv("data/readings.csv", target: "fault") -> auto_hardware
    model = classifier(in: 8, classes: 4) -> optimize(speed: max)
    while accuracy < 0.9 max_iters(5):
        train model on dataset(epochs: 2) -> self_heal
    results = run model(dataset) -> stream_output
    export model -> "v05_pipeline.ae.bin"
    save_json(results, "report.json")
```

```aether
use ai.core
use web.core

task ServeAI:
    model = load_model("model.ae.bin") -> auto_hardware
    serve model on port(8080) -> route("/predict")
```

</div>

---

## v0.8 — Packages, Planning & the OpenCL Tier

```powershell
aether init my_agent                          # manifest + src/main.ae + gitignore
cd my_agent
aether install ../shared/tools.ae             # git URLs, dirs, or single files
```

- **`aether init <name>`** — scaffolds a project: `aether.toml` manifest,
  `src/main.ae`, `README.md`, `.gitignore`, `.aether_packages/`.
- **`aether install <source>`** — installs reusable `.ae` modules from a git URL,
  a local directory, or a single `.ae` file into `.aether_packages/<name>/`.
  Import with `use pkg.<name>` — the package's top-level `fn` libraries merge into
  your program at compile time; uninstalled packages produce a clear `AE0200`
  diagnostic that names the exact fix.
- **`plan("goal") -> max_steps(3)`** — declarative multi-step agent planning
  through the LLM pipeline (mock-aware, deterministic offline), returning a
  numbered plan as a string:
  `1. Parse the goal and constraints · 2. Gather the data and context · 3. …`
- **OpenCL dispatch tier** — `--target=opencl` emits a host/kernel plan that
  loads the vendor ICD at **runtime** (LoadLibrary/dlopen — no OpenCL SDK needed
  to compile). Probing requires a *registered* ICD (Khronos registry), not just a
  stub DLL; without one you get honest fallback telemetry instead of a crash.

## v0.7 — LLM Orchestration from Machine Code

```aether
use ai.core
use data.core
use agent.core
use llm.core

task LLMAgentWorkflow:
    dataset = load_csv("data/readings.csv", target: "fault") -> auto_hardware
    model = classifier(in: 8, classes: 4) -> optimize(speed: max)
    probe = tensor(16, 16) -> auto_hardware
    while accuracy < 0.9 max_iters(4):
        train model on dataset(epochs: 2) -> self_heal
    let prev = recall("best_acc", default: 0.0)
    remember("best_acc", accuracy)
    let brief = "Fault classifier converged at accuracy " + str(accuracy) + " (previous best " + str(prev) + ")"
    let reply = ask llm(brief) -> model("gpt-4o-mini")
    log reply
    results = run model(dataset) -> stream_output
    save_json(results, "llm_report.json")
```

- **`llm.core` / `ask llm`** — compiles to raw C99 HTTP POST logic
  (Winsock2/POSIX sockets, hand-rolled JSON escaping/parsing, zero dependencies).
  Point `AETHER_LLM_URL`/`AETHER_LLM_KEY` at any OpenAI-compatible endpoint (local
  llama.cpp/Ollama works great); set `AETHER_LLM_MOCK=1` for deterministic offline
  runs — and with no URL/key configured, Aether auto-falls back to mock mode and
  *says so* in the telemetry.
- **Tensor math** — `min(t)`, `max(t)`, `argmax(t)` (flat index, returns `int`),
  and element-wise scaling `t * 0.5` — all valid inside larger expressions
  (`min(probe * 0.5)`), compiled to native C loops.
- **String building** — `+` concatenates strings, `str(value)` converts scalars,
  so prompts can embed live training metrics.
- **CUDA training kernels** — `--target=cuda` now emits real forward/backward SGD
  kernels (fused GEMM+bias+ReLU, softmax-CE, transposed GEMMs with atomic
  accumulation, SGD-momentum and gradient-clipping passes), parameterized by the
  program's own `classifier`/`train` intents — with the same honest fallback when
  nvcc is absent.

## v0.6 — Zero-Python Standalone + Persistent Agent Memory

```powershell
python build_standalone.py      # one command, PyInstaller
dist\aether.exe run hello_aether.ae
# works with Python REMOVED from PATH — compiler + NumPy + spec all bundled (19 MB)
```

- **`aether.exe`** — the whole toolchain (compiler, both backends, MCP server, spec)
  as a single native-launchable binary. Verified: stripped-PATH environment →
  `.ae` → C99 → machine code → results.
- **Persistent agent memory (`agent.core`)** —
  `remember("best_acc", accuracy)` / `let prev = recall("best_acc", default: 0.0)`.
  State lives in `<program>.memory.json` (plain JSON, identical bytes from the C99
  and Python backends) so an agent literally remembers its previous run and tries
  to beat itself. Compiled to native C99 file I/O with a hand-rolled JSON reader.
- **Tensor reductions** — `sum(results)` / `mean(results)` compile to
  `#pragma omp parallel for reduction(+:acc)` C loops.

`examples/smart_agent_memory.ae` — the self-improving agent, measured live:

```text
run 1:  recalled best_acc = 0      (default)  → trained → remembered 0.8333
run 2:  recalled best_acc = 0.8333 (hit)      → trained → remembered 0.8667
run 3:  recalled best_acc = 0.8667 (hit)      → trained → remembered 0.9034 ...
```

Each execution reads its own past, raises the bar, and writes the new best back.

## v0.5 — GPU Dispatch, Real Data, Guarded Loops

- **`auto_hardware` now probes for GPUs** — `wmic` + `nvidia-smi -L` detect NVIDIA /
  AMD / Intel hardware and `nvcc` availability. `--target=cuda` emits a real CUDA
  kernel plan (compiled by nvcc) when a GPU + toolkit exist; otherwise Aether falls
  back to the AVX-512/OpenBLAS C99 path **and says so** in the `[device]` telemetry
  and JSON `note` — it never fakes GPU execution.
- **`data.core`** — `load_csv("path.csv", target: "col")` parses real CSV files
  (numeric features, string-label interning, runtime shape discovery) and
  `save_json(value, "path.json")` writes strict JSON — both compiled to native C99
  file I/O with zero Python dependencies.
- **`while cond max_iters(N):`** — conditional loops with a built-in infinite-loop
  guard that emits a `while_guard` event instead of hanging your agent. Plus the
  built-in **`accuracy`** binding: after `train`, the best validation accuracy is
  readable in the loop condition.
- **`fn name(a, b) -> float:`** — reusable functions with typed returns, compiled to
  static C functions (or nested Python defs), with arity/type checking at compile
  time.

Measured live on the native binary (`examples/real_data_pipeline.ae`, 480-row CSV):
the `while` loop converges **0.833 → 0.867 → 0.904** accuracy and exits *before*
its guard, then writes `report.json` from pure C — whole pipeline **8.3 ms**.

## v0.4 — Turing-Complete Aether

v0.3 let agents *train and ship* models. v0.4 lets them **program**:

- **Scalars & arithmetic** — `let threshold = 0.85`, `let total = total + 2`, unary
  minus, `+ - * / %` with C-style precedence. `/` is true division on both backends.
- **Loops** — `repeat 10 times:` (integer literal or int variable), gated on
  **`agent.core`**: loops are how agents act autonomously.
- **Branches** — `if confidence > threshold:` / `else:` with full comparisons
  (`== != > < >= <=`).
- **`log "..."`** — structured output your agent can read from the run payload.
- Everything compiles to the **same native C99** — loops and branches are plain C
  `for`/`if` in the generated binary; the Python backend gets idiomatic `for`/`if`.

An autonomous retraining pipeline in 8 lines (`examples/autonomous_loop.ae`):

```aether
use ai.core
use agent.core

task AutonomousPipeline:
    dataset = load_data(samples: 5000, features: 64, classes: 5) -> auto_hardware
    model = classifier(in: 64, classes: 5) -> optimize(speed: max)
    repeat 3 times:
        train model on dataset(epochs: 2) -> self_heal
    export model -> "v04_agent.ae.bin"
```

Each loop iteration trains with self-healing and keeps the weights — measured live
on the native binary: accuracy climbs **0.844 → 0.896 → 0.941** across iterations,
then the trained agent exports itself.

## Live native inference

```powershell
aether run examples/serve_ai_api.ae          # this binary IS the server
#   [model]  loaded model.ae.bin | 128 -> 128 -> 10 | 17802 params | AEB1
#   [serve]  http://127.0.0.1:8080/predict | POST {"features":[...]} | GET /health

curl -X POST http://127.0.0.1:8080/predict -d "{\"features\": [0.01, 0.02, ...]}"
# {"prediction":6,"confidence":0.636,"probabilities":[...],"in_dim":128,"out_dim":10,"elapsed_ms":0.09}
```

Sub-0.1 ms per request, straight from machine code. `GET /health` reports model
facts; malformed requests get structured `400`s; unknown routes get `404`s.

## The number that matters

`aether bench examples/train_ai_agent.ae` runs the *identical* workload through
every implementation — Aether's native C binary, Aether's Python backend, a
hand-written NumPy script, and a hand-written PyTorch script — and reports:

```
aether bench  |  train_ai_agent.ae  |  task TrainAgent
  workload  10,000 samples x 128 features | 10 classes | hidden 128 | 5 epochs | batch 256 | lr 0.1 | 8 threads | median of 3

  Implementation                Tokens  Lines      Train      Infer      Total  Speedup  Accuracy
  aether -> C/OpenMP (native)       62      7    179.8 ms     17.1 ms    233.1 ms    1.68x    0.9050
  aether -> Python/NumPy            62      7    362.3 ms     14.6 ms    404.8 ms    0.96x    0.9218
  python/numpy (hand-written)    1,480    168    332.9 ms     14.7 ms    390.6 ms    1.00x    0.9077
  pytorch (hand-written)           937    107    374.7 ms      9.7 ms    400.9 ms    0.97x    0.9024

  token savings 62 vs 1,480 -> 96% fewer tokens; 7 lines of .ae vs 168 lines of Python
```

- **96% fewer AI tokens** than the equivalent hand-written Python (62 vs 1,480).
- **1.68× faster** than hand-written NumPy, **1.72× faster** than PyTorch (total task time).
- **7 lines vs 168** — and the 7 lines are typed by an agent, not a human.
- Measured on a shared Windows x64 machine (8C/16T, gcc 16.2 + OpenBLAS 0.3.34),
  median of 3 runs per implementation. `aether bench` reproduces this on your machine.

## 30-Second Quickstart

```powershell
git clone https://github.com/aether-ai-lang/aether.git
cd aether

# build the standalone binary (one command; or use python aether.py directly)
python build_standalone.py

# compile to a native C binary and run it — no Python needed afterwards
dist\aether.exe run hello_aether.ae

# loops, branches, arithmetic, user functions — compiled to C
dist\aether.exe run examples\control_flow.ae

# loops, branches, arithmetic, user functions — compiled to C
python aether.py run examples/control_flow.ae

# real CSV -> convergence loop -> native JSON report
python aether.py run examples/real_data_pipeline.ae

# an autonomous agent that retrains itself in a loop
python aether.py run examples/autonomous_loop.ae

# validate any .ae file — always JSON, always with a fix_patch when one exists
dist\aether.exe check hello_aether.ae

# the headline: Aether vs hand-written Python/NumPy/PyTorch, side by side
dist\aether.exe bench examples\train_ai_agent.ae
```

That's it. The standalone binary needs no Python; building it needs PyInstaller
(`python -m pip install pyinstaller`), and the Python fallback path needs only
NumPy. The native backend uses any C compiler (gcc/clang/MSVC).

<details>
<summary>Optional: put <code>aether</code> on your PATH</summary>

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\bin" | Out-Null
Copy-Item dist\aether.exe "$env:USERPROFILE\bin\aether.exe"
# then use: aether run hello_aether.ae
```

</details>

## Architecture

```
        your AI agent writes 4-7 lines of intent (.ae)  ──or──  speaks MCP
                                  |                                  |
                                  v                                  v
   +----------------------------------------------------------------+-----------+
   |                       aether compiler                          |  aether   |
   |   lexer -> parser -> type checker -> fix_patch synthesis       |  mcp      |
   |   (errors are JSON + unified diffs an agent can apply itself)  |  (stdio   |
   +----------------+------------------------------+----------------+  JSON-RPC) |
                    | --target=c  (default)        | --target=python  +-----------+
                    v                              v
   +--------------------------------+   +---------------------------+
   |  .aether_build/<name>.c        |   | .aether_build/<name>.py   |
   |  standalone C99, no runtime    |   | NumPy / OpenBLAS backend  |
   |  deps beyond libc + OpenMP     |   +-------------+-------------+
   +----------------+---------------+                 |
                    | gcc -O3 -march=native           |
                    |     -fopenmp (+OpenBLAS,        |
                    |      -lws2_32 on Windows)       |
                    v                                 |
   +--------------------------------+                 |
   |  .aether_build/<name>.exe      |                 |
   |  OpenMP threads                |                 |
   |  adaptive BLAS thread pool     |                 |
   |  fused bias+ReLU               |                 |
   |  AEB1 load / export            |                 |
   |  native HTTP/JSON server       |                 |
   +----------------+---------------+                 |
                    |                                 |
                    v                                 v
   +----------------------------------------------------------------+
   |  live chunk events | JSON payloads | HTTP | self-healing patches|
   +----------------------------------------------------------------+
```

One intent graph, two backends, byte-identical semantics. The compiler — not the
programmer, not the agent — chooses dtype, fusion, buffering, thread counts, and
even the BLAS pool size per matrix shape.

## Why agents need their own language

Python was designed for human typists. Agents pay per token and per round-trip,
and they make *systematic* mistakes that a compiler can repair mechanically.
Aether is built around three ideas:

1. **Intent, not implementation.** `train model on dataset(epochs: 5) -> self_heal`
   says everything. There is no loop to write, no dtype to pick, no allocator to tune.
2. **Errors are contracts, not crashes.** Every diagnostic has a stable code, a
   human-readable hint, and a `fix_patch` unified diff. Apply, re-check, continue —
   no model round-trip wasted on re-reading the whole file.
3. **The runtime is a machine-code artifact.** `aether run` emits C, compiles it,
   and executes the binary. Agents don't ship an interpreter; they ship a plan —
   and since v0.4 that plan can loop, branch, and do arithmetic.

## MCP: agents driving Aether natively

`aether mcp` speaks the Model Context Protocol (JSON-RPC 2.0, newline-delimited,
stdio) — plug it into OpenCode, Claude Code, Cursor, or any MCP client:

```json
{
  "mcpServers": {
    "aether": {
      "command": "python",
      "args": ["aether.py", "mcp"]
    }
  }
}
```

| Tool | What the agent can do |
|---|---|
| `aether_check` | validate any `.ae` file and read structured diagnostics |
| `aether_fix` | **auto-apply the combined self-healing patch**, then re-validate |
| `aether_compile` | emit + compile the native C binary (or Python artifact) |
| `aether_run` | execute a task and read the machine-readable result |
| `aether_spec` | pull the language spec (whole or per-section) into context |
| `aether_bench` | measure tokens / lines / speed against Python & PyTorch |

The full loop — *write → check → heal → compile → run → serve* — happens inside the
agent's tool loop, with no human keystrokes.

## The language in 5 lines (inference)

```aether
use ai.core

task HelloAether:
    data = tensor(2048, 1024) -> auto_hardware
    net = neural_net(in: data, out: 256) -> optimize(speed: max)
    run net(data) -> stream_output
```

Output (native C backend, live):

```
  [device] auto_hardware -> c/native | 16 threads | OpenBLAS | f32
  [tensor] 2048x1024 | f32 | 8.00 MiB | normal(seed=0xAE7E)
  [net]    1024 -> 512 -> 256 | 656,128 params | 2.50 MiB | optimize(speed:max)
  [stream] 8 chunks | 256 rows/chunk | out 256 | speed:max | OpenMP
  [chunk 1/8] rows 0000-0255 | mean -0.0132 | std 1.4498 | peak 5.899 | 3.62 ms
  ...
  [done]   2.68 GFLOP | 31.0 ms | 86.6 GFLOP/s | out[2048x256] | checksum -6699.69
```

## Self-healing diagnostics

`aether check` *always* answers in JSON. When a safe repair exists, the payload
carries a unified diff that fixes **all** independent problems in one `git apply`:

```json
{
  "status": "error",
  "fix_patch": "--- a/examples/self_heal.ae\n+++ b/examples/self_heal.ae\n@@ -3,2 +3,2 @@\n-use ai\n+use ai.core\n...",
  "diagnostics": [
    {
      "code": "AE0300",
      "severity": "error",
      "line": 5,
      "column": 12,
      "message": "tensor(...) requires '-> auto_hardware'",
      "hint": "Append '-> auto_hardware' so the compiler can bind a device."
    }
  ]
}
```

```powershell
python aether.py check examples/self_heal.ae   # 3 errors -> one combined patch
git apply fix.patch
python aether.py check examples/self_heal.ae   # status: ok
```

Stable codes (`AE0210` type mismatch, `AE0300` missing modifier, `AE0201` unknown
intent with "did you mean" repairs, …) are documented in
[`AETHER_SPEC.md`](AETHER_SPEC.md).

## CLI

```
aether run <file.ae>    [--target auto|c|python] [--cc PATH] [--task NAME]
                        [--chunks N] [--json] [--emit-only] [--build-dir DIR]
aether check <file.ae>  [--compact]
aether bench <file.ae>  [--task NAME] [--repeat N] [--threads N] [--json]
aether mcp              # Model Context Protocol server for AI agents
aether version
```

Exit codes: `0` success · `1` compile error · `2` usage/file error · `3` runtime error.
`--json` makes `run`, `check`, and `bench` fully machine-readable. Serving tasks
block by design — `aether run` *is* the server; Ctrl+C stops it.

## Backends

| | `--target=cuda` | `--target=c` (default) | `--target=python` |
|---|---|---|---|
| Requires | NVIDIA GPU + nvcc | any C compiler | Python 3.10+ |
| Artifact | `<name>.cu` → nvcc binary | `<name>.c` → `.exe` | `<name>.py` |
| Runtime deps | CUDA runtime | libc + OpenMP + sockets (+ OpenBLAS) | NumPy + stdlib |
| Threading | CUDA kernels | OpenMP + adaptive BLAS pool + thread-per-connection | OpenBLAS + threaded HTTP |
| Fallback | → C99 (with an explicit note) | → Python (with an explicit note) | — |

Aether finds gcc/clang/MSVC via `PATH`, `$AETHER_CC`, or a portable toolchain at
`%LOCALAPPDATA%\Aether\toolchain\w64devkit`; OpenBLAS via `$AETHER_OPENBLAS` or
`%LOCALAPPDATA%\Aether\toolchain\openblas`; GPUs via `wmic`/`nvidia-smi`/`nvcc`.
With no compiler at all, everything still runs on the Python backend — same
language, same diagnostics, same results.

## Repository layout

```
AETHER_SPEC.md               language specification (grammar, intents, error contract)
aether.py / aether.cmd       CLI entrypoints (source mode)
build_standalone.py          one-command standalone binary builder (dist/)
aetherc/                     the compiler
  lexer.py parser.py ast_nodes.py    front end (blocks, expressions, functions)
  checker.py diagnostics.py          type rules + fix_patch synthesis + patch application
  gpu.py                             GPU/CUDA probing and target dispatch
  codegen.py                         Python/NumPy backend
  cbackend.py                        C99 backend: OpenMP, OpenBLAS, HTTP, CSV/JSON, memory
  runtime.py                         NumPy runtime (devices, training, streaming, serving)
  mcp.py                             MCP server (JSON-RPC over stdio)
  bench.py bench/                    benchmark orchestration + hand-written references
hello_aether.ae              5-line inference demo
examples/train_ai_agent.ae   the training demo used by `aether bench`
examples/serve_ai_api.ae     the 4-line HTTP serving demo
examples/control_flow.ae     loops, branches, arithmetic, fn, while guards
examples/real_data_pipeline.ae  CSV -> while convergence -> native JSON report
examples/smart_agent_memory.ae  persistent memory across executions (v0.6)
examples/llm_agent_workflow.ae  CSV -> train -> LLM summary -> JSON (v0.7)
examples/agent_planner.ae    multi-step planning + training (v0.8)
examples/autonomous_loop.ae  self-retraining agent pipeline
examples/self_heal.ae        intentionally broken file for the healing demo
examples/data/readings.csv   480-row sample dataset for data.core
tests/test_aether.py         end-to-end tests (both backends, HTTP, MCP, control flow)
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

End-to-end coverage: compile and run on **both** backends, standalone-C emission,
training + AEB1 export parity, **live HTTP inference against the compiled native
server and the Python server**, **real CSV ingestion + while convergence + native
JSON output**, **persistent memory across separate executions (native + Python)**,
**mock-mode LLM orchestration on both backends**, **CUDA training-kernel emission**,
**OpenCL emit/compile/fallback**, **project init + package install + pkg import**,
agent planning on both backends, GPU probe/CUDA fallback behavior, MCP
initialize/tools/fix/compile/spec flows, patch application, control-flow semantics
(loops, branches, arithmetic, fn, guards) on both backends, JSON contracts, bench
output, and applying the combined self-heal patch with `git apply`.

## Roadmap (v0.9+)

- TLS support in the native LLM client for https:// endpoints
- Real OpenCL training kernels (forward/backward), CUDA/OpenCL device selection
- Streaming LLM completions (chunked SSE) as a `stream` intent
- A public package index (`aether install <name>` from a curated registry)
- `agent.core` planning intents: tool calls, multi-step reasoning, memory search

## License

[MIT](LICENSE) © 2026 Ermiyas — Founder of Aether

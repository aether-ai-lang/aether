<div align="center">

# ⬡ Aether (`.ae`)

### *The First Programming Language Built for AI Agents, Not Human Typists.*

You declare **what** the AI workload is. Aether decides **how** to run it — and emits
standalone, multi-threaded **C99 machine code**: training jobs, streaming inference,
native HTTP servers, GPU dispatch, and real CSV→model→JSON pipelines. The language is
Turing-complete with `while` loops and user functions, and your agent drives
everything over **MCP**.

![license](https://img.shields.io/badge/license-MIT-blue)
![backend](https://img.shields.io/badge/backend-C99%20%2B%20OpenMP%20%2B%20Winsock2-orange)
![gpu](https://img.shields.io/badge/GPU-auto%20dispatch%20%28CUDA%20%3E%20C99%20%3E%20Python%29-blueviolet)
![turing](https://img.shields.io/badge/language-Turing--complete-blueviolet)
![blas](https://img.shields.io/badge/BLAS-OpenBLAS%20%7C%20blocked--sgemm-red)
![mcp](https://img.shields.io/badge/MCP-server%20%28stdio%20JSON--RPC%29-9cf)
![python](https://img.shields.io/badge/python%20fallback-3.10%2B-informational)
![tests](https://img.shields.io/badge/tests-33%2F33-brightgreen)
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

# compile to a native C binary and run it (falls back to Python/NumPy
# automatically if no C compiler is installed)
python aether.py run hello_aether.ae

# loops, branches, arithmetic, user functions — compiled to C
python aether.py run examples/control_flow.ae

# real CSV -> convergence loop -> native JSON report
python aether.py run examples/real_data_pipeline.ae

# an autonomous agent that retrains itself in a loop
python aether.py run examples/autonomous_loop.ae

# validate any .ae file — always JSON, always with a fix_patch when one exists
python aether.py check hello_aether.ae

# the headline: Aether vs hand-written Python/NumPy/PyTorch, side by side
python aether.py bench examples/train_ai_agent.ae
```

That's it. No virtual environment, no pip installs — the Python fallback needs
only NumPy; the native backend needs any C compiler (gcc/clang/MSVC).

<details>
<summary>Optional: put <code>aether</code> on your PATH</summary>

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\bin" | Out-Null
Copy-Item aether.cmd "$env:USERPROFILE\bin\"
# then add %USERPROFILE%\bin to PATH and use: aether run hello_aether.ae
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
aether.py / aether.cmd       CLI entrypoints
aetherc/                     the compiler
  lexer.py parser.py ast_nodes.py    front end (blocks, expressions, functions)
  checker.py diagnostics.py          type rules + fix_patch synthesis + patch application
  gpu.py                             GPU/CUDA probing and target dispatch
  codegen.py                         Python/NumPy backend
  cbackend.py                        C99 backend: OpenMP, OpenBLAS, HTTP, CSV/JSON I/O
  runtime.py                         NumPy runtime (devices, training, streaming, serving)
  mcp.py                             MCP server (JSON-RPC over stdio)
  bench.py bench/                    benchmark orchestration + hand-written references
hello_aether.ae              5-line inference demo
examples/train_ai_agent.ae   the training demo used by `aether bench`
examples/serve_ai_api.ae     the 4-line HTTP serving demo
examples/control_flow.ae     loops, branches, arithmetic, fn, while guards
examples/real_data_pipeline.ae  CSV -> while convergence -> native JSON report
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
JSON output on both backends**, GPU probe/CUDA fallback behavior, MCP
initialize/tools/fix/compile/spec flows, patch application, control-flow semantics
(loops, branches, arithmetic, fn, guards) on both backends, JSON contracts, bench
output, and applying the combined self-heal patch with `git apply`.

## Roadmap (v0.6+)

- Real CUDA training (kernels for SGD backward passes, not just inference)
- AMD ROCm / OpenCL dispatch tiers behind `auto_hardware`
- Tensor arithmetic + reductions as intents (`mean`, `sum`, slicing)
- HTTPS/TLS and streaming (chunked) responses in `web.core`
- `agent.core` planning intents: memory, tool calls, multi-step reasoning

## License

[MIT](LICENSE) © 2026 Ermiyas — Founder of Aether

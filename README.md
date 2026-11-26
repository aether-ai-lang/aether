<div align="center">

# ⬡ Aether (`.ae`)

### *The First Programming Language Built for AI Agents, Not Human Typists.*

You declare **what** the AI workload is. Aether decides **how** to run it — and emits
standalone, multi-threaded **C99 machine code**. No Python at runtime. No boilerplate
for your agent to type.

![license](https://img.shields.io/badge/license-MIT-blue)
![backend](https://img.shields.io/badge/backend-C99%20%2B%20OpenMP-orange)
![blas](https://img.shields.io/badge/BLAS-OpenBLAS%20%7C%20blocked--sgemm-red)
![python](https://img.shields.io/badge/python%20fallback-3.10%2B-informational)
![tests](https://img.shields.io/badge/tests-19%2F19-brightgreen)
![tokens](https://img.shields.io/badge/tokens-96%25%20saved-success)

**Train a classifier. Export the weights. Evaluate on a stream. Seven lines.**

```aether
use ai.core

task TrainAgent:
    dataset = load_data(samples: 10000, features: 128) -> auto_hardware
    model = classifier(in: 128, classes: 10) -> optimize(speed: max)
    train model on dataset(epochs: 5) -> self_heal
    export model -> "model.ae.bin"
    run model(dataset) -> stream_output
```

</div>

---

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
git clone https://github.com/ermiyas/aether.git
cd aether

# compile to a native C binary and run it (falls back to Python/NumPy
# automatically if no C compiler is installed)
python aether.py run hello_aether.ae

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
              your AI agent writes 7 lines of intent (.ae)
                                  |
                                  v
   +----------------------------------------------------------------+
   |                       aether compiler                          |
   |   lexer -> parser -> type checker -> fix_patch synthesis       |
   |   (errors are JSON + unified diffs an agent can apply itself)  |
   +----------------+------------------------------+----------------+
                    | --target=c  (default)        | --target=python
                    v                              v
   +--------------------------------+   +---------------------------+
   |  .aether_build/train_agent.c   |   | .aether_build/train.py    |
   |  standalone C99, no runtime    |   | NumPy / OpenBLAS backend  |
   |  deps beyond libc + OpenMP     |   +-------------+-------------+
   +----------------+---------------+                 |
                    | gcc -O3 -march=native           |
                    |     -fopenmp (+OpenBLAS)        |
                    v                                 |
   +--------------------------------+                 |
   |  .aether_build/train_agent.exe |                 |
   |  OpenMP threads                |                 |
   |  adaptive BLAS thread pool     |                 |
   |  fused bias+ReLU               |                 |
   |  AEB1 model export             |                 |
   +----------------+---------------+                 |
                    |                                 |
                    v                                 v
   +----------------------------------------------------------------+
   |   live chunk events  |  JSON payloads  |  self-healing patches  |
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
   and executes the binary. Agents don't ship an interpreter; they ship a plan.

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
aether version
```

Exit codes: `0` success · `1` compile error · `2` usage/file error · `3` runtime error.
`--json` makes `run`, `check`, and `bench` fully machine-readable.

## Backends

| | `--target=c` (default) | `--target=python` |
|---|---|---|
| Artifact | `.aether_build/<name>.c` → `.exe` | `.aether_build/<name>.py` |
| Runtime deps | libc + OpenMP (+ OpenBLAS if found) | NumPy/OpenBLAS |
| Threading | OpenMP + adaptive BLAS pool | OpenBLAS |
| Streaming | native, live per-chunk output | live per-chunk output |
| Training | SGD+momentum, self-healing guards | same semantics |
| Model format | AEB1 (identical bytes) | AEB1 (identical bytes) |

Aether finds gcc/clang/MSVC via `PATH`, `$AETHER_CC`, or a portable toolchain at
`%LOCALAPPDATA%\Aether\toolchain\w64devkit`; OpenBLAS via `$AETHER_OPENBLAS` or
`%LOCALAPPDATA%\Aether\toolchain\openblas`. With no compiler at all, everything
still runs on the Python backend — same language, same diagnostics, same results.

## Repository layout

```
AETHER_SPEC.md               language specification (grammar, intents, error contract)
aether.py / aether.cmd       CLI entrypoints
aetherc/                     the compiler
  lexer.py parser.py ast_nodes.py    front end (indentation-aware, 4-space blocks)
  checker.py diagnostics.py          type rules + fix_patch synthesis
  codegen.py                         Python/NumPy backend
  cbackend.py                        C99 backend + toolchain discovery
  runtime.py                         NumPy runtime (devices, training, streaming)
  bench.py bench/                    benchmark orchestration + hand-written references
hello_aether.ae              5-line inference demo
examples/train_ai_agent.ae   the training demo used by `aether bench`
examples/self_heal.ae        intentionally broken file for the healing demo
tests/test_aether.py         19 end-to-end tests
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

19 end-to-end tests: compile and run on **both** backends, standalone-C emission,
training + AEB1 export parity, classifier diagnostics with patches, JSON contracts,
bench output, native-default detection, and applying the combined self-heal patch
with `git apply`.

## Roadmap (v0.3+)

- Loops, conditionals, and arithmetic over tensors in the intent graph
- GPU backend behind `auto_hardware` (the binding is already the language contract)
- `load` intent to read AEB1 models back into a task
- Importable intent packs beyond `ai.core`
- Deterministic cross-backend parity tests (bit-identical digests)

## License

[MIT](LICENSE) © 2026 Ermiyas — Founder of Aether

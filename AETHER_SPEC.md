# Aether (.ae) — Language Specification v0.5

**Aether is intent-driven:** you declare *what* must happen (a dataset, a model, a
training run, a stream, an HTTP endpoint); the compiler decides *how* (device, dtype,
fusion, buffering, threading, even the BLAS thread pool). v0.5 adds **GPU dispatch**
(`auto_hardware` probes NVIDIA/AMD/Intel hardware and `--target=cuda` compiles CUDA
kernels when a GPU + nvcc exist), **real data ingestion** (`data.core`: CSV in,
JSON out, compiled to native C99 file I/O), `while` loops with safety guards, and
reusable `fn` functions. Programs still compile to **standalone, multi-threaded C**
(zero Python at runtime) or optimized NumPy/OpenBLAS Python, and `aether mcp`
exposes the whole toolchain to AI agents.

## 1. Lexical rules

- Files are UTF-8, extension `.ae`, one statement per line.
- Indentation is significant. Use **4 spaces** per level; tabs are rejected (`AE0002`).
- Comments start with `#` and run to end of line.
- Identifiers: `[A-Za-z_][A-Za-z0-9_]*`. Task names are PascalCase by convention.
- Numbers: integer (`1024`) or decimal (`0.5`). Strings: `"..."` or `'...'`.
- Operators: arithmetic `+ - * / %`, comparison `== != > < >= <=`, unary `-`.

## 2. Grammar (EBNF)

```
program     := { use_decl | fn_decl | task_decl }
use_decl    := "use" IDENT { "." IDENT } NEWLINE
fn_decl     := "fn" IDENT "(" [ IDENT { "," IDENT } ] ")" "->" type ":"
               NEWLINE INDENT { stmt } DEDENT
task_decl   := "task" IDENT [ "->" modifier ] ":" NEWLINE INDENT { stmt } DEDENT
stmt        := IDENT "=" expr NEWLINE
             | "let" IDENT "=" expr NEWLINE
             | "repeat" ( NUMBER | IDENT ) "times" ":" NEWLINE INDENT { stmt } DEDENT
             | "while" expr [ "max_iters" "(" NUMBER ")" ] ":" NEWLINE INDENT { stmt } DEDENT
             | "if" expr ":" NEWLINE INDENT { stmt } DEDENT
               [ "else" ":" NEWLINE INDENT { stmt } DEDENT ]
             | "log" expr NEWLINE
             | "train" IDENT "on" IDENT "(" "epochs" ":" NUMBER ")" { "->" modifier } NEWLINE
             | "export" IDENT "->" STRING NEWLINE
             | "serve" IDENT "on" "port" "(" NUMBER ")" { "->" modifier } NEWLINE
             | "emit" expr NEWLINE
             | "return" expr NEWLINE
             | expr NEWLINE
expr        := comparison { "->" modifier }
comparison  := additive [ ("=="|"!="|">"|"<"|">="|"<=") additive ]
additive    := term { ("+"|"-") term }
term        := unary { ("*"|"/"|"%") unary }
unary       := "-" unary | primary
primary     := NUMBER | STRING | IDENT | call | "(" comparison ")"
call        := IDENT "(" [ args ] ")"
             | "run" IDENT "(" [ arg ] ")"
args        := arg { "," arg }
arg         := [ IDENT ":" ] value
value       := NUMBER | STRING | IDENT
modifier    := IDENT [ "(" [ args ] ")" ]
type        := "int" | "float" | "bool" | "str"
```

`run` is a special form: it takes a **model variable**, then the input tensor or
dataset. `serve` is a blocking statement (top-level only). Modifiers chain with `->`.

## 3. Declarations

```
use ai.core
use data.core
use web.core
use agent.core
```
`ai.core` is required before any AI intent. `data.core` before `load_csv`/`save_json`;
`web.core` before `serve`; `agent.core` before `repeat`/`while` loops. Omitting any
of them is `AE0200` with a fix patch that inserts the line. `use` lines must come
before tasks.

```
task <Name>:
    <statements>
```
A task is the unit of compilation and execution. The **first task is the default
entrypoint**. A task body must contain at least one statement. No parameters in v0.2.

## 4. Intents

| Intent | Signature | Returns | Required modifier |
|---|---|---|---|
| allocate input | `tensor(rows, cols)` | `tensor[rows, cols]` | `-> auto_hardware` |
| synthetic dataset | `load_data(samples: N, features: D, classes: K?)` | `dataset` | `-> auto_hardware` |
| CSV dataset | `load_csv("path.csv", target: "col")` | `dataset` (runtime dims) | `-> auto_hardware`, needs `data.core` |
| load model | `load_model("model.ae.bin")` | `model` (served) | `-> auto_hardware` |
| build network | `neural_net(in: X, out: Y, hidden: N?)` | `model` | none (see `optimize`) |
| build classifier | `classifier(in: D, classes: K, hidden: N?)` | `model` | none (see `optimize`) |
| training | `train model on dataset(epochs: E)` | — (sets `accuracy`) | `-> self_heal` |
| inference | `run model(data)` | `stream` | `-> stream_output` |
| JSON out | `save_json(value, "path.json")` | bytes written | needs `data.core` |
| serving | `serve model on port(P)` | — (blocks) | `-> route("/...")`, needs `web.core` |
| persistence | `export model -> "model.ae.bin"` | — | — |

### `tensor(rows, cols) -> auto_hardware`
Allocates a materialized f32 tensor of shape `[rows, cols]` filled from a seeded
normal distribution. `rows`/`cols` must be positive integer literals (1..16384,
at most 100,000,000 elements). `auto_hardware` lets the compiler bind the best
registered backend and is mandatory — omitting it is `AE0300`.

### `load_data(samples: N, features: D, classes: K?) -> auto_hardware`
Materializes a synthetic classification dataset: `N` samples (1..1,000,000),
`D` features (1..4096), `K` classes (2..1000, default 10). Labels come from a
seeded linear logit model with noise, so training has real signal
(`N*D ≤ 100,000,000`). The dataset stays on the compile-selected device.

### `neural_net(in: X, out: Y, hidden: N?) -> optimize(speed: max|balanced|low)`
Builds a dense ReLU network `in_dim -> hidden -> out_dim` with He initialization.
`in` must be a tensor; `out` is the output width; `hidden` defaults to 512.
Networks are pre-trained by construction and ready for `run`.

### `classifier(in: D, classes: K, hidden: N?) -> optimize(speed: max|balanced|low)`
Builds a trainable dense classifier `D -> hidden -> K` (hidden defaults to 128).
`in`/`classes` must match the dataset used in `train`. A missing `optimize` is a
warning (`AE0401`) and defaults to `balanced`. A task-level `-> optimize(...)`
provides the default for models in that task.

### `train model on dataset(epochs: E) -> self_heal`
Optimizes the classifier with mini-batch SGD + momentum (batch 256, lr 0.1,
gradient-norm clip 5.0). `self_heal` is mandatory (`AE0300`): after every step and
epoch the runtime validates loss/gradients, rolls back to the best checkpoint and
halves the learning rate on divergence, emitting `heal` events. `epochs` is an
integer literal 1..1000. Feature and class mismatches are compile errors (`AE0210`).

### `load_model("model.ae.bin") -> auto_hardware`
Loads a previously exported **AEB1** model file (see `export`). The path is a string
literal; relative paths resolve against the source file's directory. `auto_hardware`
is required (`AE0300`). The returned model has no compile-time shape — the loader
validates the header and dimensions at runtime — so it cannot be used with `run`
(`AE0210`); its purpose is `serve`.

### `serve model on port(P) -> route("/predict")`
Turns the program into a **blocking HTTP/JSON inference server**. Requires
`use web.core` (`AE0200`) and a `route` modifier (`AE0300`). `P` is an integer
literal 1..65535; the route must be a quoted string starting with `/`.
The `AETHER_PORT` environment variable overrides the port at launch.

The native C backend implements the server directly on Winsock2 (Windows) or POSIX
sockets, with one thread per connection; the Python backend uses the standard
library's threaded HTTP server. Behavior is identical:

| Endpoint | Behavior |
|---|---|
| `GET /health` (or `/`) | `{"status":"ok","model":"served","in_dim":D,"out_dim":K,"route":...}` |
| `POST <route>` | body `{"features":[...]}` (or a bare JSON array); returns `{"prediction":i,"confidence":p,"probabilities":[...],"in_dim":D,"out_dim":K,"elapsed_ms":ms}` |
| anything else | `404 {"error":"not found"}` |
| wrong feature count / invalid JSON | `400 {"error":"features must be an array of D numbers"}` |

### `run model(data) -> stream_output`
Chunked inference. For `neural_net` models `data` is a tensor whose width matches
the input; for classifiers `data` is a dataset, and the stream reports per-chunk
accuracy, confidence, and loss. `stream_output` is mandatory (`AE0301`): results
are emitted chunk by chunk without materializing the full output.

### `export model -> "model.ae.bin"`
Writes the trained weights in the **AEB1** format: 32-byte header
(`AEB1`, version, kind, in_dim, hidden, out_dim, classes, reserved) followed by
raw little-endian f32 `W1, b1, W2, b2`. Classifiers must be trained first
(`AE0210`). Relative paths resolve against the source file's directory. The same
format is produced by the C and Python backends.

### Control flow and scalars (v0.4+)

| Statement | Semantics |
|---|---|
| `let name = expr` | binds a scalar (`int`, `float`, `str`, or `bool`) or the result of a user `fn`. Types are stable: rebinding with a different type is `AE0210`. Tensors/datasets/models bind with plain assignment. |
| `repeat N times:` | runs the indented body `N` times; `N` is an integer literal (1..1,000,000) or an int variable. Requires `agent.core`. |
| `while cond:` / `max_iters(N)` | loops while the comparison holds. A built-in safety guard exits after `N` iterations (default 100,000) and emits a `while_guard` event — agents cannot hang forever. Requires `agent.core`. |
| `if cond:` / `else:` | runs the first branch whose comparison holds. `cond` must be a comparison (`AE0210` otherwise); `else` is optional. |
| `log expr` | emits a `log` event with the string or number. |
| `fn name(a, b) -> type:` | reusable pure function over numeric scalars; body uses `let/log/if/repeat/while/return`. At least one `return` is required, and the returned type must match the annotation. Compiled to a static C function. |
| `accuracy` (built-in) | after any `train`, `accuracy` holds the best validation accuracy seen so far — readable in `while` conditions. |

### `data.core` (v0.5)

- `load_csv("path.csv", target: "col") -> auto_hardware` — parses a CSV with a
  header row; every non-target column becomes a numeric feature, the target column
  holds class ids or string labels (interned automatically). Feature/class counts
  are discovered at runtime and validated against the model. Relative paths resolve
  against the source directory. Native C99: manual file parsing, no libc CSV deps.
- `save_json(value, "path.json")` — writes strict JSON: scalars as
  `{"name": value}`, streams (from `run`) as `{rows, accuracy, loss, gflops,
  checksum, digest}`, models as `{kind, dims, params, trained}`. Native C99:
  direct `fputs`, zero Python.
- `train` binds **`accuracy`** (float): the best validation accuracy so far —
  the natural `while accuracy < 0.9 max_iters(5):` convergence loop.

Operators: `+ - * / %` (with unary `-`) and comparisons `== != > < >= <=`.
Precedence: `%` and `*` `/` bind tighter than `+` `-`, which bind tighter than
comparisons; parentheses group. `/` is **true division** on both backends (C is
emitted with explicit double casts); `%` is integer-only. Arithmetic requires
numeric scalars — tensors, datasets, and models are not numbers (`AE0210`).

### `emit` and `return`
`emit <value>` prints a summary of a tensor, dataset, stream result, number, or
string. `return <value>` ends the task and yields the value to the host.

## 5. Types

`tensor`, `dataset`, `model`, `stream`, `int`, `float`, `str`, `bool`. Loops and
conditionals make the language Turing-complete; there are still no user-defined
functions. Invalid operations are compile-time errors (`AE0210`), never silent
coercions.

## 6. Examples

Inference (5 lines of intent):

```aether
use ai.core

task HelloAether:
    data = tensor(2048, 1024) -> auto_hardware
    net = neural_net(in: data, out: 256) -> optimize(speed: max)
    run net(data) -> stream_output
```

Training an AI agent (7 lines — dataset, model, training with self-healing,
persistence, evaluation):

```aether
use ai.core

task TrainAgent:
    dataset = load_data(samples: 10000, features: 128) -> auto_hardware
    model = classifier(in: 128, classes: 10) -> optimize(speed: max)
    train model on dataset(epochs: 5) -> self_heal
    export model -> "model.ae.bin"
    run model(dataset) -> stream_output
```

Serving a trained model as a native HTTP API (4 lines; run `aether run` and the
binary *is* the server):

```aether
use ai.core
use web.core

task ServeAI:
    model = load_model("model.ae.bin") -> auto_hardware
    serve model on port(8080) -> route("/predict")
```

An autonomous agent pipeline — loops retrain with self-healing until the
controller is satisfied (v0.4):

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

A real-data pipeline — CSV in, `while` convergence loop, JSON report out, all in
native C99 (v0.5):

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

### GPU dispatch (`auto_hardware` in v0.5)

At compile time Aether probes display/discrete GPUs (`wmic`, `nvidia-smi -L`) and
the CUDA toolkit (`nvcc`). Dispatch order for `--target auto`:

1. **CUDA** — only when an NVIDIA GPU *and* `nvcc` are both present; emits a
   `.cu` kernel plan compiled by nvcc.
2. **C99** — multi-threaded OpenMP + OpenBLAS on the detected SIMD tier
   (AVX-512 / AVX2 / NEON), with the GPU tier in the `[device]` telemetry.
3. **Python** — when no C compiler exists.

`--target=cuda` on a machine without nvcc falls back to C99 with an explicit
note in both human output and JSON (`note` field); it never silently fakes GPU
execution.

## 7. Backends and the CLI

```
aether run <file.ae>   [--target auto|c|python] [--cc PATH] [--task NAME]
                       [--chunks N] [--json] [--emit-only] [--build-dir DIR]
aether check <file.ae> [--compact]
aether bench <file.ae> [--task NAME] [--repeat N] [--threads N] [--json]
aether mcp
aether version
```

- **`auto` (default):** compiles to standalone C and executes the native binary
  whenever a C compiler (gcc/clang/MSVC) is found; otherwise falls back to the
  Python/NumPy backend with a visible notice. The native binary is written to
  `.aether_build/<name>.c` and compiled to `.aether_build/<name>.exe` with
  `-O3 -march=native -fopenmp` (plus `-lws2_32` on Windows), linking OpenBLAS when
  available. At runtime the C backend adapts the BLAS thread pool per GEMM shape
  and streams live output — **no Python involved in execution**.
- `--target=python` forces the NumPy/OpenBLAS backend (`.aether_build/<name>.py`).
- `--emit-only` writes the generated artifact without running it.
- `check` always prints JSON diagnostics; exits `0` (ok) / `1` (errors) / `2` (file).
- `bench` runs the same task through every available implementation — Aether C,
  Aether Python, a hand-written NumPy script, and a hand-written PyTorch script —
  and reports tokens, lines of code, train/infer/total time (median of `--repeat`),
  and accuracy.
- **`mcp`** runs the Model Context Protocol server (JSON-RPC 2.0 over stdio,
  newline-delimited) so AI coding agents can drive the toolchain directly:

  | Tool | Purpose |
  |---|---|
  | `aether_check` | validate an `.ae` file; returns the full diagnostics JSON |
  | `aether_fix` | return the combined `fix_patch` and (by default) apply it, then re-validate |
  | `aether_compile` | emit and compile the native C binary or Python artifact |
  | `aether_run` | execute a task, returning the machine-readable run payload |
  | `aether_spec` | read `AETHER_SPEC.md` (whole or one `##` section) |
  | `aether_bench` | run the cross-implementation benchmark |

  MCP config for agents: command `python aether.py`, args `["mcp"]`
  (OpenCode/Claude Code/Cursor all speak this protocol).
- Exit codes: `0` success, `1` compile error, `2` usage/file error, `3` runtime error.

## 8. Machine-readable errors (self-healing contract)

On any syntax, semantic, or runtime failure, agents receive JSON with `status` and,
whenever a safe repair is known, a unified-diff `fix_patch` that can be applied with
`git apply` or `patch -p1`:

```json
{
  "status": "error",
  "fix_patch": "--- a/bad.ae\n+++ b/bad.ae\n@@ -5,2 +5,2 @@\n-    model = classifier(in: 32, classes: 4) -> optimize(speed: max)\n+    model = classifier(in: 32, classes: 10) -> optimize(speed: max)\n",
  "diagnostics": [
    {
      "code": "AE0210",
      "severity": "error",
      "line": 5,
      "column": 12,
      "message": "Class mismatch: classifier has 4, dataset has 10",
      "hint": "Use classifier(in: 128, classes: 10)."
    }
  ]
}
```

Diagnostic codes: `AE0001` invalid character, `AE0002` tab indentation, `AE0003`
inconsistent dedent, `AE0100` syntax error, `AE0102` missing colon, `AE0200`
missing/unknown module, `AE0201` unknown intent, `AE0210` type mismatch, `AE0211`
undefined symbol, `AE0212` invalid argument, `AE0300`/`AE0301` missing required
modifier, `AE0302` unknown modifier, `AE0401` missing optimization policy,
`AE0500` duplicate task, `AE0501` no tasks, `AE0503` unknown task, `AE9000`
runtime failure, `AE9001` native codegen/compile failure, `AE9002` native
toolchain missing, `AE0000` unreadable/missing file, `AE9999` internal compiler error.

## 9. Agent authoring checklist

1. Start with `use ai.core`, then one `task Name:` with a 4-space indented body.
2. Valid shapes: `tensor(...) -> auto_hardware`,
   `load_data(...) -> auto_hardware`,
   `neural_net(in: x, out: N) -> optimize(speed: max)`,
   `classifier(in: D, classes: K) -> optimize(speed: max)`,
   `train model on dataset(epochs: N) -> self_heal`,
   `export model -> "model.ae.bin"`,
   `run model(data) -> stream_output`.
3. Dimensions, epochs, repeat counts, and ports are integer literals (repeat also
   takes an int variable); export paths and routes are string literals; routes must
   start with `/`.
4. Define every variable before use; `train` before `export`; `serve` is last — it
   blocks and must be top-level. `let` scalars keep one type for their whole life.
5. Run `aether check file.ae`; if `status == "error"`, apply `fix_patch` and re-check
   (or call the `aether_fix` MCP tool, which does both in one step).

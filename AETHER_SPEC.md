# Aether (.ae) — Language Specification v0.2

**Aether is intent-driven:** you declare *what* must happen (a dataset, a model, a
training run, a stream); the compiler decides *how* (device, dtype, fusion,
buffering, threading, even the BLAS thread pool). v0.2 compiles the same intent
graph to **standalone, multi-threaded C** (`--target=c`, the default when a C
compiler is present — zero Python at runtime) or to optimized NumPy/OpenBLAS Python.

## 1. Lexical rules

- Files are UTF-8, extension `.ae`, one statement per line.
- Indentation is significant. Use **4 spaces** per level; tabs are rejected (`AE0002`).
- Comments start with `#` and run to end of line.
- Identifiers: `[A-Za-z_][A-Za-z0-9_]*`. Task names are PascalCase by convention.
- Numbers: integer (`1024`) or decimal (`0.5`). Strings: `"..."` or `'...'`.

## 2. Grammar (EBNF)

```
program     := { use_decl | task_decl }
use_decl    := "use" IDENT { "." IDENT } NEWLINE
task_decl   := "task" IDENT [ "->" modifier ] ":" NEWLINE INDENT { stmt } DEDENT
stmt        := IDENT "=" expr NEWLINE
             | "train" IDENT "on" IDENT "(" "epochs" ":" NUMBER ")" { "->" modifier } NEWLINE
             | "export" IDENT "->" STRING NEWLINE
             | "emit" expr NEWLINE
             | "return" expr NEWLINE
             | expr NEWLINE
expr        := call { "->" modifier }
call        := IDENT "(" [ args ] ")"
             | "run" IDENT "(" [ arg ] ")"
args        := arg { "," arg }
arg         := [ IDENT ":" ] value
value       := NUMBER | STRING | IDENT
modifier    := IDENT [ "(" [ args ] ")" ]
```

`run` is a special form: it takes a **model variable**, then the input tensor or
dataset. Modifiers chain with `->`; at most one of each kind is meaningful.

## 3. Declarations

```
use ai.core
```
The only module in v0.2. It is required before any intent call. `use` lines must
come before tasks.

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
| load dataset | `load_data(samples: N, features: D, classes: K?)` | `dataset` | `-> auto_hardware` |
| build network | `neural_net(in: X, out: Y, hidden: N?)` | `model` | none (see `optimize`) |
| build classifier | `classifier(in: D, classes: K, hidden: N?)` | `model` | none (see `optimize`) |
| training | `train model on dataset(epochs: E)` | — | `-> self_heal` |
| inference | `run model(data)` | `stream` | `-> stream_output` |
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

### `emit` and `return`
`emit <value>` prints a summary of a tensor, dataset, stream result, number, or
string. `return <value>` ends the task and yields the value to the host.

## 5. Types

`tensor`, `dataset`, `model`, `stream`, `int`, `float`, `str`. v0.2 has no
arithmetic, conditionals, loops, or user-defined functions. Invalid operations are
compile-time errors (`AE0210`), never silent coercions.

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

## 7. Backends and the CLI

```
aether run <file.ae>   [--target auto|c|python] [--cc PATH] [--task NAME]
                       [--chunks N] [--json] [--emit-only] [--build-dir DIR]
aether check <file.ae> [--compact]
aether bench <file.ae> [--task NAME] [--repeat N] [--threads N] [--json]
aether version
```

- **`auto` (default):** compiles to standalone C and executes the native binary
  whenever a C compiler (gcc/clang/MSVC) is found; otherwise falls back to the
  Python/NumPy backend with a visible notice. The native binary is written to
  `.aether_build/<name>.c` and compiled to `.aether_build/<name>.exe` with
  `-O3 -march=native -fopenmp`, linking OpenBLAS when available. At runtime the C
  backend adapts the BLAS thread pool per GEMM shape and streams live output —
  **no Python involved in execution**.
- `--target=python` forces the NumPy/OpenBLAS backend (`.aether_build/<name>.py`).
- `--emit-only` writes the generated artifact without running it.
- `check` always prints JSON diagnostics; exits `0` (ok) / `1` (errors) / `2` (file).
- `bench` runs the same task through every available implementation — Aether C,
  Aether Python, a hand-written NumPy script, and a hand-written PyTorch script —
  and reports tokens, lines of code, train/infer/total time (median of `--repeat`),
  and accuracy.
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
3. Dimensions and epochs are integer literals; export paths are string literals.
4. Define every variable before use; `train` before `export`; tasks are straight-line.
5. Run `aether check file.ae`; if `status == "error"`, apply `fix_patch` and re-check.

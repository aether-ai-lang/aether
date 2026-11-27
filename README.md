<div align="center">

# ⬡ Aether (`.ae`)

### *The First Programming Language Built for AI Agents, Not Human Typists.*

You declare **what** the AI workload is. Aether decides **how** — and ships as a
**single standalone binary** that compiles your intent to multi-threaded C99
machine code: training, inference, HTTP servers, LLM calls, persistent memory.

![license](https://img.shields.io/badge/license-MIT-blue)
![release](https://img.shields.io/badge/release-v1.0.0-critical)
![standalone](https://img.shields.io/badge/standalone-aether.exe%20%28zero%20Python%29-critical)
![backend](https://img.shields.io/badge/backend-C99%20%2B%20OpenMP%20%2B%20Winsock2-orange)
![gpu](https://img.shields.io/badge/GPU-CUDA%20%3E%20OpenCL%20%3E%20C99%20dispatch-blueviolet)
![mcp](https://img.shields.io/badge/MCP-server%20%28stdio%20JSON--RPC%29-9cf)
![ci](https://img.shields.io/badge/CI-ubuntu%20%7C%20macos%20%7C%20windows-brightgreen)
![tests](https://img.shields.io/badge/tests-40%2B-brightgreen)

**[🌐 Landing Page](https://aether-ai-lang.github.io/aether/) ·
[📖 Spec](AETHER_SPEC.md) ·
[📦 Changelog](CHANGELOG.md) ·
[🔐 Security](SECURITY.md)**

</div>

---

## ⚡ 5-Line Quickstart

```powershell
git clone https://github.com/aether-ai-lang/aether.git
cd aether
python build_standalone.py              # one command → dist/aether.exe (19 MB, zero Python)
dist\aether.exe run hello_aether.ae     # .ae → C99 → machine code → results
dist\aether.exe bench examples\train_ai_agent.ae
```

No C compiler? Aether auto-falls back to its bundled Python/NumPy backend —
same language, same semantics, honestly reported.

## 📊 The Benchmark

`aether bench examples/train_ai_agent.ae` runs the *identical* workload through
every implementation (median of 3):

| Implementation | Tokens | Lines | Train | Infer | Total | Speedup | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| **aether → C/OpenMP (native)** | **62** | **7** | **179.8 ms** | 17.1 ms | **233.1 ms** | **1.68×** | 0.9050 |
| aether → Python/NumPy | 62 | 7 | 362.3 ms | 14.6 ms | 404.8 ms | 0.96× | 0.9218 |
| python/numpy (hand-written) | 1,480 | 168 | 332.9 ms | 14.7 ms | 390.6 ms | 1.00× | 0.9077 |
| pytorch (hand-written) | 937 | 107 | 374.7 ms | 9.7 ms | 400.9 ms | 0.97× | 0.9024 |

> **96% fewer AI tokens · 1.68× faster than hand-written NumPy · 7 lines vs 168.**

## 🏗 Architecture

```
        your AI agent writes 4-9 lines of intent (.ae)  ──or──  speaks MCP
                                  |                                  |
                                  v                                  v
   +----------------------------------------------------------------+----------+
   |                       aether compiler                          |  aether  |
   |   lexer -> parser -> type checker -> fix_patch synthesis       |  mcp     |
   +----------------+------------------------------+----------------+ (stdio   |
                    | --target=cuda / opencl / c   | --target=python|  JSON-RPC)|
                    v                              v                +----------+
   +--------------------------------+   +---------------------------+
   |  .aether_build/<name>.c / .cu  |   | .aether_build/<name>.py   |
   |  standalone C99, CUDA kernels, |   | NumPy / OpenBLAS backend  |
   |  OpenCL host (runtime ICD)     |   +-------------+-------------+
   +----------------+---------------+               |
                    | gcc/nvcc/cl -O3               |
                    v                               |
   +--------------------------------+               |
   |  machine code: OpenMP, BLAS,   |               |
   |  HTTP server, LLM client,      |               |
   |  file I/O, memory, dispatch    |               |
   +----------------+---------------+               |
                    |                               |
                    v                               v
   +----------------------------------------------------------------+
   |  live events | JSON | HTTP | self-healing patches | reports    |
   +----------------------------------------------------------------+
```

## 🌐 Full-Stack in a Few Lines

```aether
use ai.core
use data.core
use agent.core
use llm.core

task SmartAgent:
    dataset = load_csv("data/readings.csv", target: "fault") -> auto_hardware
    model = classifier(in: 8, classes: 4) -> optimize(speed: max)
    while accuracy < 0.9 max_iters(4):
        train model on dataset(epochs: 2) -> self_heal
    remember("best_acc", accuracy)
    let reply = ask llm("Fault classifier at accuracy " + str(accuracy)) -> model("gpt-4o-mini")
    log reply
    export model -> "agent.ae.bin"
```

Serve any model over HTTP:

```aether
use ai.core
use web.core

task ServeAI:
    model = load_model("model.ae.bin") -> auto_hardware
    serve model on port(8080) -> route("/predict")
```

## 🧰 What's inside

| Intent pack | Capabilities |
|---|---|
| `ai.core` | tensors, MLPs, classifiers, SGD+self-heal training, streaming inference, AEB1 export |
| `data.core` | CSV ingestion, JSON load/save — native C99 file I/O |
| `agent.core` | loops, persistent memory across executions, LLM-backed `plan()` |
| `llm.core` | OpenAI-compatible chat from raw C99 sockets (WinHTTP/libcurl), mock mode |
| `web.core` | model-as-JSON-API servers, bearer auth, 1 MiB body limit |
| `sys.core` | file read/write, JSON field loading — machine-speed glue |

**Backends:** `--target=cuda` (training kernels) → `opencl` (runtime ICD) →
`c` (OpenMP/AVX-512/OpenBLAS) → `python` — auto-dispatched, honestly reported.
**Agents:** `aether mcp` speaks the Model Context Protocol (check / self-heal /
compile / run / spec / bench). **Standalone:** `build_standalone.py` → one binary.

## 📚 Learn more

- [`AETHER_SPEC.md`](AETHER_SPEC.md) — full grammar, intents, error contract
- [`CHANGELOG.md`](CHANGELOG.md) — release-by-release history
- [`SECURITY.md`](SECURITY.md) — security model, hardening, reporting
- [`docs/index.html`](docs/index.html) — landing page (GitHub Pages)
- [`examples/`](examples/) — runnable demos for every feature

## 🧪 Tests

```powershell
python -m unittest discover -s tests -v
```

40+ end-to-end tests across both backends: HTTP inference, memory persistence,
CSV pipelines, CUDA/OpenCL emission, MCP flows, patch application, control-flow
semantics. CI runs the suite on Linux, macOS, and Windows.

## License

[MIT](LICENSE) © 2026 Ermiyas — Founder of Aether

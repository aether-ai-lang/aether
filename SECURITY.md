# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | ✅ |
| < 1.0 | ❌ (upgrade — the standalone binary makes this one command) |

## Reporting a vulnerability

Email **ermiyaschane@outlook.com** with:

1. A description of the issue and its impact.
2. Minimal reproduction (an `.ae` program, a CLI invocation, or a diff).
3. The version (`aether version`) and platform.

You will receive an acknowledgement within 72 hours and a fix timeline within
7 days. Please do not open public issues for exploitable vulnerabilities —
hardening fixes are released and credited after a patch ships.

## Aether's security model

Aether is a **compiler**, not a sandbox. Its job is to lower your declared
intent into machine code that does exactly what the program says — nothing
more, nothing less. Threat model by layer:

### Compiler (aether / aetherc)

- `.ae` programs are **data**: parsed with a recursive-descent parser that
  accepts only the documented grammar. There is no `eval`, no dynamic import
  of program text, and no shell invocation anywhere in the front end.
- Diagnostics and `fix_patch` payloads are plain JSON/strings — they never
  execute. Applying a patch is a decision the *caller* makes
  (`git apply` / the MCP `aether_fix` tool).
- `aether install` **never** invokes a shell. Git clones run with
  `subprocess.run(list_args, shell=False)`, https/SSH-URL-only schemes,
  a strict character whitelist, and rejection of leading `-` (option
  injection). Local sources are copied with `shutil`, not executed.
- Generated C is emitted from a **closed set of templates** parameterized by
  checked integer/string constants; program identifiers are emitted verbatim
  as C identifiers (validated by the lexer `[A-Za-z_][A-Za-z0-9_]*`), and all
  string literals pass through an escaping pass.

### Generated binaries (C99 backends)

- The C99 HTTP server (`web.core` / `serve`) binds **127.0.0.1** by default
  (loopback only), enforces a **1 MiB request-body hard limit** (HTTP 413),
  and supports optional bearer auth: set `AETHER_API_KEY` and every request
  must present `Authorization: Bearer <key>` (HTTP 401 otherwise).
- The LLM client (`llm.core`) reads credentials from
  `AETHER_LLM_KEY`/`AETHER_LLM_URL` environment variables — never from the
  program text. TLS on Windows uses Schannel via the OS-shipped WinHTTP; on
  POSIX it delegates to the system libcurl. The deterministic mock mode
  (`AETHER_LLM_MOCK=1`) performs no network I/O and is the default when
  credentials are absent.
- Memory limits: request buffers are fixed-capacity; model/dataset allocation
  sizes are validated at compile time and checked at load time (AEB1 headers).
- Training/export writes are confined to the paths declared in the program
  (relative paths resolve against the source directory).

### Sandboxing roadmap (v1.1+)

- [ ] `serve` bind-host declaration (`loopback | lan | any`) with deny-by-default.
- [ ] Per-task filesystem allow-lists for `sys.core` (deny by default, declared paths only).
- [ ] Manifest-level capability declarations in `aether.toml`
      (`[capabilities] network = ["api.openai.com"]`, `files = ["data/"]`), enforced
      by the compiler and auditable by reviewers.
- [ ] WASM compilation tier for in-browser/untrusted-host execution.
- [ ] Reproducible release builds + SHA-256 checksums on GitHub Releases.

### What Aether deliberately does NOT do

- It does not sandbox arbitrary C: if you ask for hardware access and your
  program declares it, the generated code does it. Review `.ae` programs the
  way you review any build script.
- The LLM client sends exactly the prompt the program constructs to the
  endpoint the operator configures. Aether adds no telemetry and no
  phones-home behavior; model replies are returned to the program, not logged
  anywhere persistent.

## Verification

- `python -m unittest discover -s tests` — the end-to-end suite includes the
  HTTP auth/limit paths via generated binaries.
- `python aether.py run examples/serve_ai_api.ae` + `curl` exercises: valid
  predictions, 401 without a key (when `AETHER_API_KEY` is set), 413 for
  oversized bodies, 404 for unknown routes, 400 for malformed features.

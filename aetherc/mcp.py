"""`aether mcp` — an AI-agent-native MCP server (JSON-RPC 2.0 over stdio).

Exposes the Aether toolchain to coding agents (OpenCode, Claude Code, Cursor):
inspect the language spec, validate .ae files, auto-apply self-healing patches,
compile native binaries, run tasks, and benchmark — all without leaving the
agent's tool loop.

Framing: newline-delimited JSON (MCP stdio transport). LSP-style
Content-Length framing is auto-detected for older clients.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .bootstrap import entry_command, resource
from .checker import Checker
from .diagnostics import AETHER_VERSION, apply_patch, merged_fix_patch
from .lexer import tokenize
from .parser import parse
from .cli import analyze, build, error_payload

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_VERSION = "2024-11-05"

SPEC = resource("AETHER_SPEC.md")
SERVER_INFO = {"name": "aether", "version": AETHER_VERSION}


# ----------------------------------------------------------------- tool impls
def _analyze_file(path: str) -> tuple[object, dict]:
    result = analyze(Path(path))
    payload = error_payload(Path(path), result, compact=False)
    return result, payload


def tool_check(args: dict) -> dict:
    file = args.get("file")
    if not file:
        return _error_result("missing required argument: file")
    _, payload = _analyze_file(str(file))
    return _text_result(json.dumps(payload, indent=2))


def tool_fix(args: dict) -> dict:
    file = args.get("file")
    if not file:
        return _error_result("missing required argument: file")
    apply = bool(args.get("apply", True))
    path = Path(file)
    result, payload = _analyze_file(str(path))
    patch = payload.get("fix_patch")
    response: dict = {
        "file": str(path),
        "before": payload.get("summary"),
        "diagnostics": payload.get("diagnostics", []),
        "fix_patch": patch,
        "applied": False,
    }
    if patch and apply:
        source = path.read_text(encoding="utf-8-sig")
        patched = apply_patch(patch, source)
        if patched is None:
            response["error"] = "patch does not apply cleanly (source may have changed)"
            return _text_result(json.dumps(response, indent=2), is_error=True)
        path.write_text(patched, encoding="utf-8", newline="\n")
        response["applied"] = True
        _, after = _analyze_file(str(path))
        response["after"] = after.get("summary")
        response["status"] = after.get("status")
    elif patch:
        response["hint"] = "call again with apply=true to write the fix"
    return _text_result(json.dumps(response, indent=2))


def tool_compile(args: dict) -> dict:
    file = args.get("file")
    if not file:
        return _error_result("missing required argument: file")
    target = args.get("target") or "auto"
    path = Path(file)
    result, payload = _analyze_file(str(path))
    if payload.get("status") == "error":
        return _text_result(json.dumps(payload, indent=2), is_error=True)
    program = result["program"]

    from .cbackend import (
        CBackendError,
        Toolchain,
        compile_c,
        find_toolchain,
        generate_for_toolchain,
    )

    toolchain = None
    if target in ("c", "auto"):
        try:
            toolchain = find_toolchain()
        except Exception:  # pragma: no cover - discovery guard
            toolchain = None

    try:
        if toolchain is not None:
            source = generate_for_toolchain(program, path.name, toolchain, None, chunks=8)
            native = compile_c(source, path.parent / ".aether_build", path.stem, toolchain)
            payload = {
                "status": "ok",
                "target": "c",
                "artifact": str(native.source_file),
                "binary": str(native.binary),
                "compile_ms": native.compile_ms,
                "compiler": toolchain.version,
                "blas": toolchain.blas_name,
            }
        else:
            if target == "c":
                return _error_result("no C compiler found; pass target='python'")
            _, generated, build_ms, _ = build(program, path, None)
            payload = {
                "status": "ok",
                "target": "python",
                "artifact": str(path.parent / ".aether_build" / f"{path.stem}.py"),
                "build_ms": build_ms,
                "lines": len(generated.splitlines()),
            }
        return _text_result(json.dumps(payload, indent=2))
    except CBackendError as exc:
        return _error_result(f"native compilation failed: {exc}")


def tool_run(args: dict) -> dict:
    file = args.get("file")
    if not file:
        return _error_result("missing required argument: file")
    command = entry_command("run", str(file), "--json")
    if args.get("task"):
        command += ["--task", str(args["task"])]
    if args.get("target"):
        command += ["--target", str(args["target"])]
    if args.get("chunks"):
        command += ["--chunks", str(int(args["chunks"]))]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=int(args.get("timeout", 120)),
        )
    except subprocess.TimeoutExpired:
        return _error_result(
            "run timed out (serving tasks block by design; kill or lower the timeout)"
        )
    try:
        payload = json.loads(proc.stdout)
    except ValueError:
        return _error_result(f"run produced no JSON: {(proc.stderr or proc.stdout)[:400]}")
    return _text_result(json.dumps(payload, indent=2), is_error=payload.get("status") != "ok")


def tool_spec(args: dict) -> dict:
    try:
        text = SPEC.read_text(encoding="utf-8-sig")
    except OSError as exc:
        return _error_result(f"cannot read spec: {exc}")
    section = args.get("section")
    if section:
        marker = f"## {section}"
        index = text.find(marker)
        if index < 0:
            sections = [
                line[3:]
                for line in text.splitlines()
                if line.startswith("## ")
            ]
            return _error_result(f"no section '{section}'; sections: {', '.join(sections)}")
        nxt = text.find("\n## ", index + 1)
        text = text[index : nxt if nxt > 0 else len(text)].strip()
    return _text_result(text)


def tool_bench(args: dict) -> dict:
    file = args.get("file")
    if not file:
        return _error_result("missing required argument: file")
    command = entry_command(
        "bench",
        str(file),
        "--json",
        "--repeat",
        str(int(args.get("repeat", 1))),
    )
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, cwd=str(ROOT), timeout=900
        )
    except subprocess.TimeoutExpired:
        return _error_result("bench timed out")
    try:
        payload = json.loads(proc.stdout)
    except ValueError:
        return _error_result(f"bench produced no JSON: {(proc.stderr or proc.stdout)[:400]}")
    return _text_result(json.dumps(payload, indent=2), is_error=payload.get("status") != "ok")


TOOLS = [
    {
        "name": "aether_check",
        "description": (
            "Validate an .ae file. Returns JSON diagnostics with stable codes, hints, "
            "and fix_patch unified diffs (exit-code semantics of `aether check`)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"file": {"type": "string", "description": "path to a .ae file"}},
            "required": ["file"],
        },
    },
    {
        "name": "aether_fix",
        "description": (
            "Self-heal an .ae file: returns the combined fix_patch and (by default) "
            "applies it to the file, then re-validates."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "apply": {"type": "boolean", "default": True},
            },
            "required": ["file"],
        },
    },
    {
        "name": "aether_compile",
        "description": (
            "Compile an .ae file to a native C binary (default when a compiler exists) "
            "or to Python. Returns artifact and binary paths."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "target": {"type": "string", "enum": ["auto", "c", "python"]},
            },
            "required": ["file"],
        },
    },
    {
        "name": "aether_run",
        "description": (
            "Compile and execute an .ae task (JSON payload). Serving tasks block by "
            "design; they will hit the timeout."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "task": {"type": "string"},
                "target": {"type": "string", "enum": ["auto", "c", "python"]},
                "chunks": {"type": "integer"},
                "timeout": {"type": "integer", "default": 120},
            },
            "required": ["file"],
        },
    },
    {
        "name": "aether_spec",
        "description": "Read AETHER_SPEC.md, or one section such as 'Intents' or 'Grammar (EBNF)'.",
        "inputSchema": {
            "type": "object",
            "properties": {"section": {"type": "string"}},
        },
    },
    {
        "name": "aether_bench",
        "description": (
            "Benchmark an .ae training task against hand-written Python/NumPy/PyTorch "
            "and report tokens, lines, and timings."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "repeat": {"type": "integer", "default": 1},
            },
            "required": ["file"],
        },
    },
]


def _text_result(text: str, is_error: bool = False) -> dict:
    result: dict = {"content": [{"type": "text", "text": text}]}
    if is_error:
        result["isError"] = True
    return result


def _error_result(message: str) -> dict:
    return _text_result(json.dumps({"status": "error", "message": message}), is_error=True)


# ------------------------------------------------------------------ transport
def _read_message(reader) -> dict | None:
    line = reader.readline()
    if not line:
        return None
    stripped = line.strip()
    if not stripped:
        return {}
    if stripped.lower().startswith(b"content-length:"):
        try:
            length = int(stripped.split(b":", 1)[1].strip())
        except ValueError:
            return {}
        while True:
            header = reader.readline()
            if header in (b"\r\n", b"\n", b""):
                break
        body = reader.read(length)
        if not body:
            return None
        return json.loads(body.decode("utf-8"))
    return json.loads(stripped.decode("utf-8"))


def _handle(request: dict) -> dict | None:
    method = request.get("method")
    request_id = request.get("id")
    is_notification = request_id is None

    if method == "initialize":
        requested = (request.get("params") or {}).get("protocolVersion")
        version = requested if isinstance(requested, str) else PROTOCOL_VERSION
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        }
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": TOOLS},
        }
    if method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        handler = {
            "aether_check": tool_check,
            "aether_fix": tool_fix,
            "aether_compile": tool_compile,
            "aether_run": tool_run,
            "aether_spec": tool_spec,
            "aether_bench": tool_bench,
        }.get(name)
        if handler is None:
            if is_notification:
                return None
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": f"unknown tool '{name}'"},
            }
        try:
            result = handler(arguments)
        except Exception as exc:  # tool-level guard: never crash the server
            result = _error_result(f"{type(exc).__name__}: {exc}")
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    if method in ("shutdown", "exit"):
        if method == "shutdown" and not is_notification:
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "resources/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "resources": [
                    {
                        "uri": "file://AETHER_SPEC.md",
                        "name": "Aether Language Specification",
                        "mimeType": "text/markdown",
                    }
                ]
            },
        }
    if is_notification:
        return None
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def run_mcp() -> int:
    """Serve MCP over stdio until stdin closes or the client exits."""
    reader = sys.stdin.buffer
    writer = sys.stdout.buffer

    def send(message: dict) -> None:
        writer.write((json.dumps(message) + "\n").encode("utf-8"))
        writer.flush()

    while True:
        try:
            request = _read_message(reader)
        except (ValueError, json.JSONDecodeError) as exc:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"parse error: {exc}"},
                }
            )
            continue
        if request is None:
            break
        if not request:
            continue
        try:
            response = _handle(request)
        except Exception as exc:  # pragma: no cover - server guard
            request_id = request.get("id")
            if request_id is not None:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"},
                    }
                )
            continue
        if response is not None:
            send(response)
    return 0

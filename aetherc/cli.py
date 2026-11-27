"""Aether command line interface: compile, check, and run .ae programs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .checker import Checker
from .codegen import generate
from .diagnostics import AETHER_VERSION, AetherError, Diagnostic, merged_fix_patch
from .lexer import tokenize
from .parser import parse
from .project import find_packages_dir, init_project, install_package, package_functions

BOLD = "\x1b[1m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
MAGENTA = "\x1b[35m"
RESET = "\x1b[0m"


class UI:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def paint(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"{code}{text}{RESET}"


def configure_threads() -> None:
    """Set BLAS thread counts before NumPy is imported."""
    threads = os.environ.get("AETHER_THREADS") or str(os.cpu_count() or 1)
    os.environ.setdefault("AETHER_THREADS", threads)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, threads)


def _merge_package_functions(program, source_path: Path) -> None:
    """Import top-level `fn` libraries from installed packages (use pkg.<name>)."""
    packages_dir = find_packages_dir(source_path)
    if packages_dir is None:
        return
    wanted = {
        use.module.split(".", 1)[1]
        for use in program.uses
        if use.module.startswith("pkg.")
    }
    existing = {fn.name for fn in program.functions}
    for package in sorted(wanted):
        package_dir = packages_dir / package
        if not package_dir.is_dir():
            continue
        for ae_file in package_functions(package_dir):
            try:
                source = ae_file.read_text(encoding="utf-8-sig")
                package_program = parse(tokenize(source, str(ae_file)), str(ae_file), source)
            except AetherError:
                continue
            for fn in package_program.functions:
                if fn.name in existing:
                    continue
                program.functions.append(fn)
                existing.add(fn.name)


def analyze(path: Path) -> dict:
    result: dict = {
        "program": None,
        "source": None,
        "diagnostics": [],
        "tokens": 0,
        "parse_ms": 0.0,
        "check_ms": 0.0,
    }

    try:
        source = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        result["diagnostics"] = [
            Diagnostic(
                "AE0000",
                f"File not found: {path}",
                hint="Pass the path to an existing .ae file.",
            )
        ]
        return result
    except OSError as exc:
        result["diagnostics"] = [
            Diagnostic("AE0000", f"Cannot read {path}: {exc}", hint="Check file permissions.")
        ]
        return result

    result["source"] = source
    started = time.perf_counter()
    try:
        tokens = tokenize(source, str(path))
        result["tokens"] = len(tokens)
        program = parse(tokens, str(path), source)
    except AetherError as exc:
        result["parse_ms"] = (time.perf_counter() - started) * 1000.0
        result["diagnostics"] = [exc.diagnostic]
        return result
    except Exception as exc:  # pragma: no cover - compiler bug guard
        result["parse_ms"] = (time.perf_counter() - started) * 1000.0
        result["diagnostics"] = [
            Diagnostic(
                "AE9999",
                f"Internal compiler error: {type(exc).__name__}: {exc}",
                hint="Please report this with the source file.",
            )
        ]
        return result

    result["parse_ms"] = (time.perf_counter() - started) * 1000.0
    result["program"] = program
    _merge_package_functions(program, path)

    started = time.perf_counter()
    try:
        result["diagnostics"] = Checker(program).check()
    except Exception as exc:  # pragma: no cover - compiler bug guard
        result["diagnostics"] = [
            Diagnostic(
                "AE9999",
                f"Internal checker error: {type(exc).__name__}: {exc}",
                hint="Please report this with the source file.",
            )
        ]
    result["check_ms"] = (time.perf_counter() - started) * 1000.0
    return result


def build(program, path: Path, build_dir: str | None) -> tuple[Path, str, float, object]:
    started = time.perf_counter()
    generated = generate(program, path.name)
    out_dir = Path(build_dir) if build_dir else path.parent / ".aether_build"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{path.stem}.py"
    out_file.write_text(generated, encoding="utf-8", newline="\n")
    code = compile(generated, str(out_file), "exec")
    build_ms = (time.perf_counter() - started) * 1000.0
    return out_file, generated, build_ms, code


def error_payload(path: Path, result: dict, compact: bool = False) -> dict:
    diagnostics: list[Diagnostic] = result["diagnostics"]
    errors = sum(1 for d in diagnostics if d.severity == "error")
    warnings = sum(1 for d in diagnostics if d.severity == "warning")
    status = "error" if errors else ("warning" if warnings else "ok")

    payload: dict = {"status": status}
    patch = None
    if result["source"] is not None:
        patch = merged_fix_patch(str(path), result["source"].splitlines(), diagnostics)
    if patch is None:
        patch = next(
            (d.fix_patch for d in diagnostics if d.severity == "error" and d.fix_patch), None
        )
    if patch is None:
        patch = next((d.fix_patch for d in diagnostics if d.fix_patch), None)
    if patch:
        payload["fix_patch"] = patch

    program = result["program"]
    payload.update(
        {
            "aether": AETHER_VERSION,
            "file": str(path),
            "summary": {
                "tasks": len(program.tasks) if program else 0,
                "errors": errors,
                "warnings": warnings,
                "tokens": result["tokens"],
            },
            "diagnostics": [d.to_dict() for d in diagnostics],
        }
    )
    return payload


def print_json(payload: dict, compact: bool = False) -> None:
    if compact:
        print(json.dumps(payload, separators=(",", ":")))
    else:
        print(json.dumps(payload, indent=2))


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def print_human_diagnostics(ui: UI, diagnostics: list[Diagnostic]) -> None:
    for diagnostic in diagnostics:
        color = RED if diagnostic.severity == "error" else YELLOW
        head = ui.paint(color, diagnostic.severity.upper())
        print(
            f"  {head:<7} {diagnostic.code}  line {diagnostic.line}:{diagnostic.column}  "
            f"{diagnostic.message}"
        )
        if diagnostic.hint:
            print(f"          hint: {diagnostic.hint}")


def make_printer(ui: UI):
    def printer(event: dict) -> None:
        kind = event["event"]
        if kind == "device":
            gpu_part = (
                f" | gpu: {event['gpu']} ({event.get('gpu_tier', 'cpu')})"
                if event.get("gpu")
                else " | gpu: none detected (cpu tier)"
            )
            if event.get("backend") == "c":
                print(
                    f"  [device] auto_hardware -> {event['device']}/native | "
                    f"{event['threads']} threads | {event.get('compiler', 'c')} | "
                    f"{event.get('blas', 'sgemm')} | {event['dtype']}"
                    f"{gpu_part}"
                )
            else:
                features = "+".join(event["features"]) or "baseline"
                print(
                    f"  [device] auto_hardware -> {event['device']}/{event['machine']} | "
                    f"{features} | {event['threads']} threads | {event['blas']} | "
                    f"{event['dtype']}{gpu_part}"
                )
        elif kind == "tensor":
            print(
                f"  [tensor] {event['rows']}x{event['cols']} | f32 | {event['mib']:.2f} MiB | "
                f"normal(seed=0x{event.get('seed', 0xAE7E):X})"
            )
        elif kind == "dataset":
            origin = (
                f" | from {Path(event['source']).name}"
                if event.get("source")
                else " | synthetic"
            )
            print(
                f"  [dataset] {event['samples']:,} samples x {event['features']} features | "
                f"{event['classes']} classes | {event['mib']:.2f} MiB{origin}"
            )
        elif kind == "model":
            label = "model" if event.get("kind") == "classifier" else "net"
            print(
                f"  [{label}]{' ' * (4 - len(label))} "
                f"{'classifier ' if event.get('kind') == 'classifier' else ''}"
                f"{event['in_dim']} -> {event['hidden']} -> {event['out_dim']} | "
                f"{event['params']:,} params | {event['mib']:.2f} MiB | "
                f"optimize(speed:{event.get('policy', 'max')})"
            )
        elif kind == "train_start":
            print(
                f"  [train] {event['epochs']} epochs | {event['samples']:,} samples | "
                f"batch {event['batch']} | {event.get('optimizer', 'sgd+momentum')} "
                f"lr={event.get('lr', 0.1):g} | self_heal"
            )
        elif kind == "epoch":
            print(
                f"  [epoch {event['epoch']}/{event['total']}] loss {event['loss']:.4f} | "
                f"acc {event['accuracy']:.4f} | lr {event['lr']:g} | {event['ms']:.1f} ms"
                + (f" | {event['heals']} heal(s)" if event.get("heals") else "")
            )
        elif kind == "heal":
            print(
                f"  [heal]   epoch {event.get('epoch', '?')} | {event['reason']} | "
                f"lr -> {event.get('lr', 0):g}"
            )
        elif kind == "while_guard":
            print(f"  [guard]  {event.get('message', 'loop guard tripped')}")
        elif kind == "remember":
            previous = (
                f" (was {event['previous']:.4g})" if event.get("previous") is not None else ""
            )
            print(f"  [memory] remembered {event['key']} = {event['value']:.4g}{previous}")
        elif kind == "recall":
            origin = "hit" if event.get("found") else "default"
            print(
                f"  [memory] recalled {event['key']} = {event['value']:.4g} ({origin})"
            )
        elif kind == "reduce":
            print(
                f"  [{event['op']:<4}]   {event['value']:.6g} | {event['elements']:,} elements"
            )
        elif kind == "scale":
            print(
                f"  [scale]  x{event['factor']:g} | out[{event['rows']}x{event['cols']}]"
            )
        elif kind == "ask":
            print(
                f"  [llm]    {event['model']} | {event['prompt_chars']} chars in | "
                f"{event['reply_chars']} chars out | {event['mode']} | {event['ms']:.1f} ms"
            )
        elif kind == "read_file":
            print(f"  [read]   {Path(event['path']).name} | {event['chars']} chars")
        elif kind == "write_file":
            print(f"  [write]  {Path(event['path']).name} | {event['bytes']} bytes")
        elif kind == "load_json":
            print(
                f"  [json]   {Path(event['path']).name}"
                + (f"['{event['key']}']" if event.get("key") else "")
                + f" = {event['value']:.6g}"
            )
        elif kind == "save_json":
            print(
                f"  [json]   {Path(event['path']).name} | {event['bytes']} bytes"
            )
        elif kind == "train_done":
            print(
                f"  [trained] acc {event['accuracy']:.4f} | loss {event['loss']:.4f} | "
                f"{event['epochs']} epochs | {event['steps']} steps | "
                f"{event.get('heals', 0)} heal(s) | {event['ms']:.1f} ms | "
                f"{event.get('samples_per_s', 0) / 1000:,.0f}k samples/s"
            )
        elif kind == "export":
            print(
                f"  [export] {Path(event['path']).name} | {event['bytes'] / 1024:.1f} KiB | "
                f"{event.get('format', 'AEB1')} | {event.get('params', 0):,} params"
            )
        elif kind == "stream_start":
            policy = event.get("policy", "balanced")
            print(
                f"  [stream] {event['chunks']} chunks | {event['chunk_rows']} rows/chunk | "
                f"out {event['out_dim']} | speed:{policy}, "
                f"work {event.get('working_rows', event['chunk_rows'])} rows"
                + (" | classifier" if event.get("kind") == "classifier" else "")
            )
        elif kind == "chunk" and "accuracy" in event:
            width = max(4, len(str(event["row_end"])))
            print(
                f"  [chunk {event['index']}/{event['total']}] "
                f"samples {event['row_start']:0{width}d}-{event['row_end']:0{width}d} | "
                f"acc {event['accuracy']:.4f} | conf {event['confidence']:.4f} | "
                f"loss {event['loss']:.4f} | {event['ms']:.2f} ms"
            )
        elif kind == "chunk":
            width = max(4, len(str(event["row_end"])))
            print(
                f"  [chunk {event['index']}/{event['total']}] "
                f"rows {event['row_start']:0{width}d}-{event['row_end']:0{width}d} | "
                f"mean {event['mean']:+.4f} | std {event['std']:.4f} | "
                f"peak {event['peak']:.3f} | {event['ms']:.2f} ms"
            )
        elif kind == "summary" and "accuracy" in event:
            print(
                f"  [done]   acc {event['accuracy']:.4f} | loss {event['loss']:.4f} | "
                f"{event['ms']:.1f} ms | {event['gflops_per_s']:.1f} GFLOP/s | "
                f"out[{event['rows']}x{event['out_dim']}] | digest {event['digest']}"
            )
        elif kind == "summary":
            print(
                f"  [done]   {event['gflops']:.2f} GFLOP | {event['ms']:.1f} ms | "
                f"{event['gflops_per_s']:.1f} GFLOP/s | out[{event['rows']}x{event['out_dim']}] | "
                f"checksum {event['checksum']:+.2f} | digest {event['digest']}"
            )
        elif kind == "print":
            print(f"  [emit]   {event['text']}")
        elif kind == "log":
            print(f"  [log]    {event['text']}")
        elif kind == "emit_tensor":
            print(
                f"  [emit]   tensor[{event['rows']}x{event['cols']}] mean={event['mean']:+.4f} "
                f"std={event['std']:.4f} peak={event['peak']:.3f}"
            )
        elif kind == "emit_dataset":
            print(
                f"  [emit]   dataset[{event['samples']}x{event['features']}] "
                f"classes={event['classes']}"
            )
        elif kind == "emit_stream":
            accuracy = (
                f" acc={event['accuracy']:.4f}" if "accuracy" in event and event["accuracy"] else ""
            )
            print(
                f"  [emit]   stream[{event['rows']}x{event['out_dim']}] "
                f"checksum={event['checksum']:+.2f} digest={event['digest']}{accuracy}"
            )
        elif kind == "serve_start":
            print(
                f"  [serve]  http://127.0.0.1:{event['port']}{event['route']} | "
                f"POST {{\"features\":[...]}} | GET /health"
            )
        elif kind == "request" and "prediction" in event:
            print(
                f"  [request] POST {event['path']} | pred {event['prediction']} "
                f"({event['confidence']:.3f}) | {event['ms']:.2f} ms"
            )
        elif kind == "serve_stop":
            print(f"  [serve]  stopped")
        sys.stdout.flush()

    return printer


def cmd_check(args: argparse.Namespace) -> int:
    path = Path(args.file)
    result = analyze(path)
    print_json(error_payload(path, result, compact=args.compact), compact=args.compact)
    if not path.is_file():
        return 2
    return 1 if any(d.severity == "error" for d in result["diagnostics"]) else 0


def cmd_run(args: argparse.Namespace) -> int:
    path = Path(args.file)
    result = analyze(path)
    diagnostics: list[Diagnostic] = result["diagnostics"]
    errors = [d for d in diagnostics if d.severity == "error"]

    if errors or not path.is_file():
        print_json(error_payload(path, result, compact=False))
        return 1 if path.is_file() else 2

    ui = UI(sys.stdout.isatty() and not args.no_color and not os.environ.get("NO_COLOR"))
    program = result["program"]
    warnings = [d for d in diagnostics if d.severity == "warning"]
    json_mode = args.json

    task_name = args.task or (program.tasks[0].name if program.tasks else None)
    if task_name is None:
        print_json(error_payload(path, result, compact=False))
        return 1

    task_names = [task.name for task in program.tasks]
    if task_name not in task_names:
        payload = {
            "status": "error",
            "code": "AE0503",
            "message": f"Unknown task '{task_name}'",
            "hint": f"Available tasks: {', '.join(task_names)}",
        }
        print_json(payload)
        return 1

    target = getattr(args, "target", "python") or "python"
    note = None
    # Inspecting generated CUDA never needs nvcc; emit it before dispatch.
    if target == "cuda" and args.emit_only:
        return _run_cuda(args, path, result, program, task_name, warnings, ui, None)
    if target == "opencl":
        return _run_opencl(args, path, result, program, task_name, warnings, ui)
    if target in ("c", "auto", "cuda"):
        from .gpu import dispatch_target

        resolved, note = dispatch_target(target)
        target = resolved

    toolchain = None
    if target == "cuda":
        return _run_cuda(args, path, result, program, task_name, warnings, ui, note)
    if target in ("c", "auto"):
        from .cbackend import CBackendError, find_toolchain

        try:
            toolchain = find_toolchain(getattr(args, "cc", None))
        except CBackendError as exc:
            if target == "c" and json_mode:
                print_json(
                    {
                        "status": "error",
                        "code": "AE9002",
                        "message": str(exc),
                        "hint": "Install gcc, clang, or MSVC, or pass --cc <path>.",
                    }
                )
                return 2
            toolchain = None

    if note and not json_mode:
        print(f"  {ui.paint(YELLOW, '[note]   ' + note)}")

    from .gpu import probe_gpu

    os.environ.setdefault("AETHER_DEVICE_TIER", str(probe_gpu().get("tier") or "cpu"))

    use_native = toolchain is not None and target in ("c", "auto")
    if not json_mode:
        print(f"{ui.paint(BOLD, 'aether')} {AETHER_VERSION}  {ui.paint(DIM, '|')}  {path.name}")
        print()
        print(f"  {'parse':<7} {result['parse_ms']:6.2f} ms   {result['tokens']} tokens")
        status = "ok" if not warnings else f"ok with {len(warnings)} warning(s)"
        task_word = "task" if len(program.tasks) == 1 else "tasks"
        print(
            f"  {'check':<7} {result['check_ms']:6.2f} ms   {status} | "
            f"{len(program.tasks)} {task_word}"
        )

    if use_native:
        return _run_native(args, path, result, program, task_name, warnings, ui, toolchain, note)

    if target == "c" and not json_mode:
        print_human_diagnostics(
            ui,
            [
                Diagnostic(
                    "AE9002",
                    "No native C compiler found; falling back to the Python backend",
                    hint="Install gcc/clang/MSVC or set AETHER_CC to the compiler path.",
                )
            ],
        )
        print()

    try:
        out_file, generated, build_ms, code = build(program, path, args.build_dir)
    except Exception as exc:  # pragma: no cover - codegen guard
        payload = {
            "status": "error",
            "code": "AE9999",
            "message": f"Code generation failed: {type(exc).__name__}: {exc}",
            "hint": "Please report this with the source file.",
        }
        print_json(payload)
        return 1

    if json_mode:
        events: list[dict] = []
        on_event = events.append
    else:
        print(f"  {'build':<7} {build_ms:6.2f} ms   {display_path(out_file)}")
        print(f"  {'target':<7} python | NumPy/OpenBLAS runtime")
        if warnings:
            print()
            print_human_diagnostics(ui, warnings)
        print()
        print(f"  {'task':<7} {task_name}")
        print()
        on_event = make_printer(ui)

    if args.emit_only:
        if json_mode:
            print_json(
                {
                    "status": "ok",
                    "aether": AETHER_VERSION,
                    "file": str(path),
                    "target": "python",
                    "artifact": str(out_file),
                    "build_ms": build_ms,
                    "tasks": [task.name for task in program.tasks],
                }
            )
        else:
            print(f"  compiled -> {display_path(out_file)}")
            print()
        return 0

    from .runtime import Runtime  # deferred so thread env is set before NumPy loads

    runtime = Runtime(chunks=args.chunks, on_event=on_event, base_dir=path.parent)
    namespace: dict = {}
    exec(code, namespace)

    tasks = namespace["build"](runtime)
    started = time.perf_counter()
    try:
        tasks[task_name](runtime)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        payload = {
            "status": "error",
            "code": "AE9000",
            "message": f"Runtime failure in task '{task_name}': {type(exc).__name__}: {exc}",
            "hint": "Check tensor shapes and that every model input matches the data width.",
            "task": task_name,
            "elapsed_ms": elapsed_ms,
        }
        if os.environ.get("AETHER_DEBUG"):
            payload["traceback"] = traceback.format_exc()
        print_json(payload)
        return 3

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if json_mode:
        events.append(
            {"event": "task_done", "task": task_name, "ms": elapsed_ms, "backend": "python"}
        )
        summary = next(
            (e for e in reversed(events) if e["event"] == "summary"),
            None,
        )
        payload = {
            "status": "ok",
            "aether": AETHER_VERSION,
            "file": str(path),
            "task": task_name,
            "target": "python",
            "note": note,
            "compile": {
                "tokens": result["tokens"],
                "parse_ms": result["parse_ms"],
                "check_ms": result["check_ms"],
                "build_ms": build_ms,
                "artifact": str(out_file),
            },
            "diagnostics": [d.to_dict() for d in diagnostics],
            "events": events,
            "result": summary,
            "elapsed_ms": elapsed_ms,
        }
        print_json(payload)
    else:
        print()
        print(f"  {ui.paint(DIM, f'{elapsed_ms:.1f} ms wall')}")

    return 0


def _run_cuda(
    args: argparse.Namespace,
    path: Path,
    result: dict,
    program,
    task_name: str,
    warnings: list[Diagnostic],
    ui: UI,
    note: str | None,
) -> int:
    """Compile and run the CUDA plan; never silently fakes GPU execution."""
    from .cbackend import CBackendError, compile_cuda, generate_cuda, run_binary
    from .gpu import probe_gpu

    json_mode = args.json
    if not json_mode:
        print(f"{ui.paint(BOLD, 'aether')} {AETHER_VERSION}  {ui.paint(DIM, '|')}  {path.name}")
        print()
        print(f"  {'parse':<7} {result['parse_ms']:6.2f} ms   {result['tokens']} tokens")
        gpu = probe_gpu()
        print(
            f"  {'device':<7} cuda | {gpu.get('name') or 'NVIDIA GPU'} | "
            f"nvcc: {gpu.get('nvcc')}"
        )
        if note:
            print(f"  {ui.paint(YELLOW, '[note]   ' + note)}")

    try:
        source = generate_cuda(program, path.name)
        out_dir = Path(args.build_dir) if args.build_dir else path.parent / ".aether_build"
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.emit_only:
            cu_file = out_dir / f"{path.stem}.cu"
            cu_file.write_text(source, encoding="utf-8", newline="\n")
            if json_mode:
                print_json(
                    {
                        "status": "ok",
                        "aether": AETHER_VERSION,
                        "file": str(path),
                        "target": "cuda",
                        "artifact": str(cu_file),
                    }
                )
            else:
                print(f"  emitted -> {display_path(cu_file)}")
            return 0
        binary, compile_ms, _command = compile_cuda(source, out_dir, path.stem)
    except CBackendError as exc:
        payload = {
            "status": "error",
            "code": "AE9003",
            "message": f"CUDA backend unavailable: {exc}",
            "hint": "Install the CUDA toolkit (nvcc) and a CUDA-capable GPU, "
            "or run with --target=c / --target=python.",
        }
        print_json(payload)
        return 1

    if args.emit_only:
        return 0

    if json_mode:
        started = time.perf_counter()
        proc = run_binary(binary, task_name, events=True, silent=True, base_dir=str(path.parent.resolve()))
        wall_ms = (time.perf_counter() - started) * 1000.0
        events = parse_events(proc.stderr)
        if proc.returncode != 0:
            print_json(
                {
                    "status": "error",
                    "code": "AE9000",
                    "message": f"CUDA task '{task_name}' failed with exit code {proc.returncode}",
                }
            )
            return 3
        print_json(
            {
                "status": "ok",
                "aether": AETHER_VERSION,
                "file": str(path),
                "task": task_name,
                "target": "cuda",
                "events": events,
                "elapsed_ms": wall_ms,
            }
        )
        return 0

    started = time.perf_counter()
    sys.stdout.flush()
    proc = run_binary(binary, task_name, events=False, silent=False, inherit=True, base_dir=str(path.parent.resolve()))
    wall_ms = (time.perf_counter() - started) * 1000.0
    print()
    if proc.returncode != 0:
        print_json(
            {
                "status": "error",
                "code": "AE9000",
                "message": f"CUDA task '{task_name}' failed with exit code {proc.returncode}",
            }
        )
        return 3
    print(f"  {ui.paint(DIM, f'{wall_ms:.1f} ms wall (cuda)')}")
    return 0


def _run_opencl(
    args: argparse.Namespace,
    path: Path,
    result: dict,
    program,
    task_name: str,
    warnings: list[Diagnostic],
    ui: UI,
) -> int:
    """Compile the OpenCL plan (runtime-loaded ICD) and run it."""
    from .cbackend import (
        CBackendError,
        compile_opencl,
        generate_opencl,
        parse_events,
        run_binary,
    )
    from .gpu import probe_gpu

    json_mode = args.json
    if not json_mode:
        print(f"{ui.paint(BOLD, 'aether')} {AETHER_VERSION}  {ui.paint(DIM, '|')}  {path.name}")
        print()
        print(f"  {'parse':<7} {result['parse_ms']:6.2f} ms   {result['tokens']} tokens")
        gpu = probe_gpu()
        print(
            f"  {'target':<7} opencl | runtime-loaded ICD | "
            f"detected: {'yes' if gpu.get('opencl') else 'no ICD found (runtime will report fallback)'}"
        )

    out_dir = Path(args.build_dir) if args.build_dir else path.parent / ".aether_build"
    try:
        host_source, kernel_source = generate_opencl(program, path.name)
        opencl_binary, opencl_compile_ms = compile_opencl(
            host_source, kernel_source, out_dir, path.stem, _toolchain_for_opencl()
        )
    except CBackendError as exc:
        payload = {
            "status": "error",
            "code": "AE9004",
            "message": f"OpenCL backend unavailable: {exc}",
            "hint": "Verify a C compiler is installed, or run with --target=c / --target=python.",
        }
        print_json(payload)
        return 1

    artifact = out_dir / f"{path.stem}_opencl.cl"
    if args.emit_only:
        if json_mode:
            print_json(
                {
                    "status": "ok",
                    "aether": AETHER_VERSION,
                    "file": str(path),
                    "target": "opencl",
                    "artifact": str(artifact),
                    "host": str(opencl_binary),
                }
            )
        else:
            print(f"  {'host':<7} {display_path(opencl_binary)}")
            print(f"  {'kernel':<7} {display_path(artifact)}")
        return 0

    # No working ICD detected: report the fallback honestly instead of
    # executing against a possibly-stub OpenCL.dll (which can crash).
    if not probe_gpu().get("opencl"):
        fallback_events = [
            {
                "event": "device",
                "device": "opencl-probe",
                "tier": "opencl-unavailable",
            },
            {
                "event": "fallback",
                "message": "no OpenCL runtime found (install an ICD from your GPU vendor)",
                "next": "dispatching semantics to the C99/AVX-512 tier (--target=c)",
            },
        ]
        if json_mode:
            print_json(
                {
                    "status": "ok",
                    "aether": AETHER_VERSION,
                    "file": str(path),
                    "task": task_name,
                    "target": "opencl",
                    "events": fallback_events,
                    "result": None,
                }
            )
        else:
            for event in fallback_events:
                if event["event"] == "device":
                    print("  [device] auto_hardware -> opencl probe | runtime-loaded ICD")
                else:
                    print(f"  [fallback] {event['message']}")
                    print(f"  [fallback] {event['next']}")
        return 0

    if json_mode:
        started = time.perf_counter()
        proc = run_binary(opencl_binary, task_name, events=True, silent=True, base_dir=str(path.parent.resolve()))
        wall_ms = (time.perf_counter() - started) * 1000.0
        events = parse_events(proc.stderr)
        print_json(
            {
                "status": "ok" if proc.returncode == 0 else "error",
                "aether": AETHER_VERSION,
                "file": str(path),
                "task": task_name,
                "target": "opencl",
                "events": events,
                "elapsed_ms": wall_ms,
            }
        )
        return 0 if proc.returncode == 0 else 3

    started = time.perf_counter()
    sys.stdout.flush()
    proc = run_binary(opencl_binary, task_name, events=False, silent=False, inherit=True, base_dir=str(path.parent.resolve()))
    wall_ms = (time.perf_counter() - started) * 1000.0
    print()
    if proc.returncode != 0:
        print_json(
            {
                "status": "error",
                "code": "AE9000",
                "message": f"OpenCL task '{task_name}' failed with exit code {proc.returncode}",
            }
        )
        return 3
    print(f"  {ui.paint(DIM, f'{wall_ms:.1f} ms wall (opencl)')}")
    return 0


def _toolchain_for_opencl():
    from .cbackend import CBackendError, find_toolchain

    toolchain = find_toolchain()
    if toolchain is None:
        raise CBackendError("no C compiler found for the OpenCL host program")
    return toolchain


def _run_native(
    args: argparse.Namespace,
    path: Path,
    result: dict,
    program,
    task_name: str,
    warnings: list[Diagnostic],
    ui: UI,
    toolchain,
    note: str | None = None,
) -> int:
    from .cbackend import (
        CBackendError,
        compile_c,
        generate_for_toolchain,
        parse_events,
        run_binary,
    )

    json_mode = args.json
    out_dir = Path(args.build_dir) if args.build_dir else path.parent / ".aether_build"
    warnings = list(warnings)

    if not json_mode:
        if warnings:
            print()
            print_human_diagnostics(ui, warnings)

    try:
        source = generate_for_toolchain(
            program, path.name, toolchain, task_names=None, chunks=args.chunks
        )
    except CBackendError as exc:
        payload = {
            "status": "error",
            "code": "AE9001",
            "message": f"Native code generation failed: {exc}",
            "hint": "The task uses an intent the C backend does not support yet.",
        }
        print_json(payload)
        return 1

    if args.emit_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        c_file = out_dir / f"{path.stem}.c"
        c_file.write_text(source, encoding="utf-8", newline="\n")
        if json_mode:
            print_json(
                {
                    "status": "ok",
                    "aether": AETHER_VERSION,
                    "file": str(path),
                    "target": "c",
                    "artifact": str(c_file),
                    "tasks": [task.name for task in program.tasks],
                }
            )
        else:
            print(f"  {'build':<7} {display_path(c_file)}")
            print(f"  {'target':<7} c | {toolchain.blas_name} | OpenMP")
            print()
            print(f"  emitted -> {display_path(c_file)}")
            print()
        return 0

    try:
        native = compile_c(source, out_dir, path.stem, toolchain)
    except CBackendError as exc:
        payload = {
            "status": "error",
            "code": "AE9001",
            "message": "Native compilation failed",
            "detail": str(exc)[-4000:],
            "hint": "Check the compiler installation, or run with --target=python.",
        }
        print_json(payload)
        return 1

    if json_mode:
        started = time.perf_counter()
        try:
            proc = run_binary(
                native.binary,
                task_name,
                events=True,
                silent=True,
                base_dir=str(path.parent.resolve()),
            )
        except subprocess.TimeoutExpired:
            payload = {
                "status": "error",
                "code": "AE9000",
                "message": f"Native task '{task_name}' timed out",
                "task": task_name,
            }
            print_json(payload)
            return 3
        wall_ms = (time.perf_counter() - started) * 1000.0
        events = parse_events(proc.stderr)
        if proc.returncode != 0:
            payload = {
                "status": "error",
                "code": "AE9000",
                "message": f"Native task '{task_name}' failed with exit code {proc.returncode}",
                "task": task_name,
                "stderr": (proc.stderr or "")[-2000:],
            }
            print_json(payload)
            return 3
        summary = next((e for e in reversed(events) if e["event"] == "summary"), None)
        payload = {
            "status": "ok",
            "aether": AETHER_VERSION,
            "file": str(path),
            "task": task_name,
            "target": "c",
            "note": note,
            "compile": {
                "tokens": result["tokens"],
                "parse_ms": result["parse_ms"],
                "check_ms": result["check_ms"],
                "build_ms": native.compile_ms,
                "artifact": str(native.source_file),
                "binary": str(native.binary),
                "compiler": toolchain.version,
                "blas": toolchain.blas_name,
            },
            "diagnostics": [d.to_dict() for d in result["diagnostics"]],
            "events": events,
            "result": summary,
            "elapsed_ms": wall_ms,
        }
        print_json(payload)
        return 0

    print(f"  {'build':<7} {native.compile_ms:6.2f} ms   {display_path(native.binary)}")
    print(
        f"  {'target':<7} c | {Path(toolchain.compiler).name} | "
        f"{toolchain.blas_name} | OpenMP | zero Python at runtime"
    )
    print()
    print(f"  {'task':<7} {task_name}")
    print()

    started = time.perf_counter()
    sys.stdout.flush()  # keep parent header ahead of the child's live output
    proc = run_binary(
        native.binary,
        task_name,
        events=False,
        silent=False,
        inherit=True,
        base_dir=str(path.parent.resolve()),
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    print()
    if proc.returncode != 0:
        payload = {
            "status": "error",
            "code": "AE9000",
            "message": f"Native task '{task_name}' failed with exit code {proc.returncode}",
            "task": task_name,
        }
        print_json(payload)
        return 3
    print(f"  {ui.paint(DIM, f'{wall_ms:.1f} ms wall (native)')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aether",
        description=(
            "Aether 1.0.0 - intent-driven AI-agent-native compiler "
            "(sys.core, HTTPS LLM client, packages, GPU dispatch, MCP)"
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"aether {AETHER_VERSION}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="compile and execute a .ae program")
    run_parser.add_argument("file", help="path to a .ae source file")
    run_parser.add_argument("--task", help="task to execute (default: first task)")
    run_parser.add_argument(
        "--chunks", type=int, default=8, help="number of streaming chunks (default: 8)"
    )
    run_parser.add_argument(
        "--target",
        choices=("auto", "c", "cuda", "opencl", "python"),
        default="auto",
        help=(
            "execution backend: auto-dispatch to CUDA/OpenCL when available, "
            "else native C (default), else Python"
        ),
    )
    run_parser.add_argument("--cc", help="path to the C compiler for --target=c")
    run_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    run_parser.add_argument(
        "--emit-only", action="store_true", help="compile the execution plan without running"
    )
    run_parser.add_argument("--build-dir", help="directory for generated artifacts")
    run_parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    run_parser.set_defaults(func=cmd_run)

    check_parser = subparsers.add_parser(
        "check", help="validate a .ae program and print JSON diagnostics"
    )
    check_parser.add_argument("file", help="path to a .ae source file")
    check_parser.add_argument("--compact", action="store_true", help="single-line JSON output")
    check_parser.set_defaults(func=cmd_check)

    bench_parser = subparsers.add_parser(
        "bench",
        help="benchmark a .ae program against an equivalent Python/PyTorch implementation",
    )
    bench_parser.add_argument("file", help="path to a .ae source file")
    bench_parser.add_argument("--task", help="task to benchmark (default: first task)")
    bench_parser.add_argument(
        "--repeat", type=int, default=3, help="runs per implementation (median reported)"
    )
    bench_parser.add_argument(
        "--threads", type=int, help="threads for every implementation (default: all cores)"
    )
    bench_parser.add_argument("--cc", help="path to the C compiler for the native backend")
    bench_parser.add_argument("--build-dir", help="directory for generated artifacts")
    bench_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    bench_parser.set_defaults(func=_cmd_bench)

    mcp_parser = subparsers.add_parser(
        "mcp",
        help="run the Aether MCP server (JSON-RPC over stdio) for AI coding agents",
    )
    mcp_parser.set_defaults(func=_cmd_mcp)

    init_parser = subparsers.add_parser(
        "init", help="scaffold a new Aether project (manifest + src/main.ae)"
    )
    init_parser.add_argument("name", help="project name (also the directory name)")
    init_parser.set_defaults(func=_cmd_init)

    install_parser = subparsers.add_parser(
        "install",
        help="install a reusable .ae module package into .aether_packages/",
    )
    install_parser.add_argument(
        "source", help="git URL, local directory of .ae modules, or a single .ae file"
    )
    install_parser.add_argument(
        "--project", help="project directory (default: current directory)"
    )
    install_parser.set_defaults(func=_cmd_install)

    version_parser = subparsers.add_parser("version", help="print the compiler version")
    version_parser.set_defaults(func=lambda args: _print_version())

    return parser


def _cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_bench

    return run_bench(args)


def _cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp import run_mcp

    return run_mcp()


def _cmd_init(args: argparse.Namespace) -> int:
    try:
        root = init_project(args.name, Path.cwd())
    except (ValueError, FileExistsError) as exc:
        print_json({"status": "error", "code": "AE0212", "message": str(exc)})
        return 2
    print(f"initialized project {args.name}  |  aether {AETHER_VERSION}")
    print(f"  {root / 'aether.toml'}")
    print(f"  {root / 'src' / 'main.ae'}   <- entry program")
    print(f"  {root / 'README.md'}")
    print(f"  {root / '.gitignore'}")
    print(f"  {root / '.aether_packages'}")
    print()
    print("next:")
    print(f"  cd {args.name}")
    print(f"  aether run src/main.ae")
    return 0


def _cmd_install(args: argparse.Namespace) -> int:
    root = Path(args.project).resolve() if args.project else Path.cwd()
    try:
        name, how = install_package(args.source, root)
    except (ValueError, FileExistsError, RuntimeError) as exc:
        print_json({"status": "error", "code": "AE0212", "message": str(exc)})
        return 2
    installed = root / ".aether_packages" / name
    print(f"installed {name}  ({how})")
    print(f"  -> {installed}")
    print(f"import it with:  use pkg.{name}")
    return 0


def _print_version() -> int:
    from .cbackend import find_toolchain

    toolchain = None
    try:
        toolchain = find_toolchain()
    except Exception:  # pragma: no cover - discovery guard
        toolchain = None
    native = f" | native C: {toolchain.kind} ({toolchain.blas_name})" if toolchain else ""
    print(f"aether {AETHER_VERSION} (numpy/OpenBLAS + native C backends){native}")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_threads()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # pragma: no cover - top-level guard
        payload = {
            "status": "error",
            "code": "AE9999",
            "message": f"Unhandled {type(exc).__name__}: {exc}",
            "hint": "Set AETHER_DEBUG=1 and re-run for a traceback.",
        }
        if os.environ.get("AETHER_DEBUG"):
            payload["traceback"] = traceback.format_exc()
        print_json(payload)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

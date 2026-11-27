"""`aether bench`: Aether vs. hand-written Python/NumPy/PyTorch, side by side.

Runs the same task through every available implementation and reports tokens,
lines of code, and execution time. The .ae program is the input for the Aether
rows; the reference rows run standalone scripts from benchmarks/ that implement
the identical workload by hand.
"""

from __future__ import annotations

import io
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .ast_nodes import Call, Literal
from .bootstrap import entry_command
from .cli import analyze, print_json
from .diagnostics import AETHER_VERSION
from .lexer import tokenize

ROOT = Path(__file__).resolve().parent.parent
STRUCTURAL = {"NEWLINE", "INDENT", "DEDENT", "EOF"}
PY_STRUCTURAL = {
    "ENCODING",
    "NL",
    "NEWLINE",
    "INDENT",
    "DEDENT",
    "ENDMARKER",
    "COMMENT",
}
# Benchmark children run at high priority so background desktop apps do not
# skew one implementation more than another.
_CHILD_FLAGS = {"creationflags": 0x00000080} if os.name == "nt" else {}


@dataclass
class Workload:
    samples: int
    features: int
    classes: int
    hidden: int
    epochs: int
    batch: int = 256
    lr: float = 0.1

    def describe(self) -> str:
        return (
            f"{self.samples:,} samples x {self.features} features | {self.classes} classes | "
            f"hidden {self.hidden} | {self.epochs} epochs | batch {self.batch} | lr {self.lr:g}"
        )


@dataclass
class Row:
    name: str
    tokens: int | None
    lines: int | None
    ms: float | None = None
    train_ms: float | None = None
    infer_ms: float | None = None
    accuracy: float | None = None
    note: str = ""


def _significant_tokens(source: str) -> int:
    return sum(1 for token in tokenize(source, "<bench>") if token.kind not in STRUCTURAL)


def _code_lines(source: str) -> int:
    return sum(
        1
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def _python_metrics(path: Path) -> tuple[int, int]:
    import tokenize as pytokenize

    text = path.read_text(encoding="utf-8-sig")
    count = 0
    try:
        for token in pytokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in (
                pytokenize.COMMENT,
                pytokenize.NL,
                pytokenize.NEWLINE,
                pytokenize.INDENT,
                pytokenize.DEDENT,
                pytokenize.ENDMARKER,
                pytokenize.ENCODING,
            ):
                continue
            count += 1
    except Exception:  # pragma: no cover - malformed reference script
        count = sum(1 for line in text.splitlines() if line.strip())
    return count, _code_lines(text)


def _bind(call: Call, names: list[str]) -> dict[str, object]:
    bound: dict[str, object] = {}
    position = 0
    for arg in call.args:
        if arg.name is None:
            if position < len(names):
                bound[names[position]] = arg.value
                position += 1
        else:
            bound[arg.name] = arg.value
    return bound


def _int_arg(value: object, default: int | None = None) -> int:
    if isinstance(value, Literal):
        return int(value.value)
    if default is not None:
        return default
    raise ValueError("expected an integer literal")


def extract_workload(program, task_name: str) -> Workload:
    task = next((t for t in program.tasks if t.name == task_name), None)
    if task is None:
        raise ValueError(f"unknown task '{task_name}'")

    samples = features = classes = hidden = epochs = None
    for stmt in task.body:
        call = stmt.expr if isinstance(stmt.expr, Call) else None
        if stmt.kind == "assign" and call is not None:
            if call.name == "load_data":
                bound = _bind(call, ["samples", "features", "classes"])
                samples = _int_arg(bound.get("samples"))
                features = _int_arg(bound.get("features"))
                classes = _int_arg(bound.get("classes"), 10)
            elif call.name == "classifier":
                bound = _bind(call, ["in", "classes", "hidden"])
                hidden = _int_arg(bound.get("hidden"), 128)
                if classes is None:
                    classes = _int_arg(bound.get("classes"))
        elif stmt.kind == "train":
            epochs = stmt.epochs

    missing = [
        name
        for name, value in (
            ("load_data(...)", samples),
            ("classifier(in: ..., hidden: ...)", hidden),
            ("train ... (epochs: N)", epochs),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "bench needs a training workload; the task is missing " + ", ".join(missing)
        )
    if features is None or classes is None:
        raise ValueError("bench could not infer the dataset shape")

    return Workload(
        samples=samples,
        features=features,
        classes=classes,
        hidden=hidden,
        epochs=epochs,
    )


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _default_threads() -> int:
    """Prefer physical cores: oversubscribing SMT threads hurts BLAS latency."""
    return _default_threads_cached()


@lru_cache(maxsize=1)
def _default_threads_cached() -> int:
    try:
        output = subprocess.run(
            ["wmic", "cpu", "get", "NumberOfCores"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        for line in output.splitlines():
            line = line.strip()
            if line.isdigit():
                return int(line)
    except Exception:
        pass
    logical = os.cpu_count() or 1
    return min(logical, 8) if logical > 8 else logical


def _task_env(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["AETHER_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env.pop("AETHER_EVENTS", None)
    env.pop("AETHER_SILENT", None)
    return env


def run_aether_python(path: Path, task: str, threads: int, repeat: int) -> list[dict]:
    env = _task_env(threads)
    runs: list[dict] = []
    for _ in range(repeat):
        proc = subprocess.run(
            entry_command(
                "run", str(path), "--task", task, "--target", "python", "--json"
            ),
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=env,
            timeout=900,
            **_CHILD_FLAGS,
        )
        try:
            payload = json.loads(proc.stdout)
        except ValueError as exc:
            raise RuntimeError(f"aether (python) produced no JSON: {proc.stdout[:400]}") from exc
        if payload.get("status") != "ok":
            raise RuntimeError(payload.get("message", "aether (python) run failed"))
        runs.append(_extract(payload["events"], payload["elapsed_ms"]))
    return runs


def _extract(events: list[dict], total_ms: float) -> dict:
    train = next((e for e in reversed(events) if e["event"] == "train_done"), None)
    infer = next((e for e in reversed(events) if e["event"] == "summary"), None)
    return {
        "ms": total_ms,
        "train_ms": train.get("ms") if train else None,
        "infer_ms": infer.get("ms") if infer else None,
        "accuracy": train.get("accuracy") if train else None,
        "steps": train.get("steps") if train else None,
    }


def run_aether_native(
    path: Path,
    program,
    task: str,
    threads: int,
    repeat: int,
    toolchain,
    build_dir: Path | None,
) -> tuple[list[dict], object]:
    from .cbackend import compile_c, generate_for_toolchain, run_binary

    os.environ["AETHER_THREADS"] = str(threads)
    source = generate_for_toolchain(program, path.name, toolchain, None, chunks=8)
    out_dir = build_dir or path.parent / ".aether_build"
    native = compile_c(source, out_dir, path.stem, toolchain)

    runs: list[dict] = []
    for _ in range(repeat):
        proc = run_binary(
            native.binary,
            task,
            events=True,
            silent=True,
            high_priority=True,
            base_dir=str(path.parent.resolve()),
        )
        events = [
            json.loads(line[len("AETHER ") :])
            for line in (proc.stderr or "").splitlines()
            if line.startswith("AETHER ")
        ]
        if proc.returncode != 0:
            raise RuntimeError(f"native binary failed ({proc.returncode})")
        done = next((e for e in events if e["event"] == "task_done"), None)
        if done is None:
            raise RuntimeError("native binary did not report task_done")
        runs.append(_extract(events, done.get("ms", 0.0)))
    return runs, native


def run_reference(
    script: Path, workload: Workload, threads: int, repeat: int, pythonpath: str | None
) -> list[dict]:
    env = _task_env(threads)
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    args = [
        "--samples",
        str(workload.samples),
        "--features",
        str(workload.features),
        "--classes",
        str(workload.classes),
        "--hidden",
        str(workload.hidden),
        "--epochs",
        str(workload.epochs),
        "--batch",
        str(workload.batch),
        "--lr",
        str(workload.lr),
    ]
    runs: list[dict] = []
    for _ in range(repeat):
        proc = subprocess.run(
            [sys.executable, str(script), *args],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=env,
            timeout=900,
            **_CHILD_FLAGS,
        )
        try:
            runs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"{script.name} failed: {(proc.stderr or proc.stdout)[:400]}") from exc
    return runs


def _torch_pythonpath() -> str | None:
    candidates = []
    explicit = os.environ.get("AETHER_TORCH_PATH")
    if explicit:
        candidates.append(Path(explicit))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "Aether" / "torch")
    for candidate in candidates:
        if (candidate / "torch" / "__init__.py").is_file():
            return str(candidate)
    return None


def _median_row(runs: list[dict]) -> dict:
    def med(key: str) -> float | None:
        values = [run[key] for run in runs if run.get(key) is not None]
        return _median(values) if values else None

    return {
        "ms": med("ms"),
        "train_ms": med("train_ms"),
        "infer_ms": med("infer_ms"),
        "accuracy": med("accuracy"),
    }


def run_bench(args) -> int:
    path = Path(args.file)
    result = analyze(path)
    errors = [d for d in result["diagnostics"] if d.severity == "error"]
    if errors or not path.is_file():
        print_json(
            {
                "status": "error",
                "code": errors[0].code if errors else "AE0000",
                "message": errors[0].message if errors else f"File not found: {path}",
            }
        )
        return 1 if path.is_file() else 2

    program = result["program"]
    task = args.task or program.tasks[0].name
    try:
        workload = extract_workload(program, task)
    except ValueError as exc:
        print_json({"status": "error", "code": "AE0503", "message": str(exc)})
        return 1

    repeat = max(1, int(args.repeat))
    threads = args.threads or _default_threads()
    source = result["source"]
    aether_tokens = _significant_tokens(source)
    aether_lines = _code_lines(source)

    rows: list[Row] = []
    native_info: dict = {}

    # ---- Aether, native C backend ------------------------------------------
    toolchain = None
    try:
        from .cbackend import find_toolchain

        toolchain = find_toolchain(args.cc)
    except Exception as exc:  # pragma: no cover - discovery guard
        toolchain = None
        native_info["error"] = str(exc)

    if toolchain is not None:
        try:
            started = time.perf_counter()
            native_runs, native = run_aether_native(
                path, program, task, threads, repeat, toolchain, args.build_dir
            )
            native_info = {
                "compiler": toolchain.version,
                "blas": toolchain.blas_name,
                "compile_ms": (time.perf_counter() - started) * 1000.0,
                "binary": str(native.binary),
            }
            med = _median_row(native_runs)
            rows.append(
                Row(
                    name="aether -> C/OpenMP (native)",
                    tokens=aether_tokens,
                    lines=aether_lines,
                    **med,
                )
            )
        except Exception as exc:
            rows.append(
                Row(
                    name="aether -> C/OpenMP (native)",
                    tokens=aether_tokens,
                    lines=aether_lines,
                    note=f"unavailable: {exc}",
                )
            )
    else:
        rows.append(
            Row(
                name="aether -> C/OpenMP (native)",
                tokens=aether_tokens,
                lines=aether_lines,
                note="no C compiler found",
            )
        )

    # ---- Aether, Python backend ---------------------------------------------
    skip_python = bool(os.environ.get("AETHER_BENCH_SKIP_PYTHON"))
    if skip_python:
        rows.append(
            Row(
                name="aether -> Python/NumPy",
                tokens=aether_tokens,
                lines=aether_lines,
                note="skipped (AETHER_BENCH_SKIP_PYTHON)",
            )
        )
    else:
        try:
            med = _median_row(run_aether_python(path, task, threads, repeat))
            rows.append(
                Row(
                    name="aether -> Python/NumPy",
                    tokens=aether_tokens,
                    lines=aether_lines,
                    **med,
                )
            )
        except Exception as exc:
            rows.append(
                Row(
                    name="aether -> Python/NumPy",
                    tokens=aether_tokens,
                    lines=aether_lines,
                    note=f"failed: {exc}",
                )
            )

    # ---- hand-written references --------------------------------------------
    skip_refs = bool(os.environ.get("AETHER_BENCH_SKIP_REFERENCES"))
    numpy_tokens, numpy_lines = _python_metrics(ROOT / "benchmarks" / "reference_numpy.py")
    if skip_refs:
        rows.append(
            Row(
                name="python/numpy (hand-written)",
                tokens=numpy_tokens,
                lines=numpy_lines,
                note="skipped (AETHER_BENCH_SKIP_REFERENCES)",
            )
        )
        baseline = rows[-1]
    else:
        try:
            med = _median_row(
                run_reference(
                    ROOT / "benchmarks" / "reference_numpy.py", workload, threads, repeat, None
                )
            )
            rows.append(
                Row(
                    name="python/numpy (hand-written)",
                    tokens=numpy_tokens,
                    lines=numpy_lines,
                    **med,
                )
            )
            baseline = rows[-1]
        except Exception as exc:
            rows.append(
                Row(
                    name="python/numpy (hand-written)",
                    tokens=numpy_tokens,
                    lines=numpy_lines,
                    note=f"failed: {exc}",
                )
            )
            baseline = None

    torch_path = _torch_pythonpath()
    torch_available = torch_path is not None and not skip_refs
    if torch_available:
        torch_tokens, torch_lines = _python_metrics(
            ROOT / "benchmarks" / "reference_torch.py"
        )
        try:
            med = _median_row(
                run_reference(
                    ROOT / "benchmarks" / "reference_torch.py",
                    workload,
                    threads,
                    repeat,
                    torch_path,
                )
            )
            rows.append(
                Row(
                    name="pytorch (hand-written)",
                    tokens=torch_tokens,
                    lines=torch_lines,
                    **med,
                )
            )
        except Exception as exc:
            rows.append(
                Row(
                    name="pytorch (hand-written)",
                    tokens=torch_tokens,
                    lines=torch_lines,
                    note=f"failed: {exc}",
                )
            )

    if args.json:
        payload = {
            "status": "ok",
            "aether": AETHER_VERSION,
            "file": str(path),
            "task": task,
            "workload": workload.__dict__,
            "threads": threads,
            "repeat": repeat,
            "native": native_info,
            "rows": [
                {**row.__dict__, "note": row.note or None} for row in rows
            ],
        }
        print_json(payload)
        return 0

    _print_table(path, task, workload, threads, repeat, rows, native_info, aether_tokens, torch_available)
    return 0


def _print_table(
    path: Path,
    task: str,
    workload: Workload,
    threads: int,
    repeat: int,
    rows: list[Row],
    native_info: dict,
    aether_tokens: int,
    torch_available: bool,
) -> None:
    DIM = "\x1b[2m"
    BOLD = "\x1b[1m"
    GREEN = "\x1b[32m"
    RESET = "\x1b[0m"
    color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    def paint(code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if color else text

    baseline = next((row for row in rows if row.name.startswith("python/numpy")), None)
    baseline_ms = baseline.ms if baseline and baseline.ms else None

    print(
        f"{paint(BOLD, 'aether bench')}  {paint(DIM, '|')}  {path.name}  "
        f"{paint(DIM, '|')}  task {task}"
    )
    print(f"  workload  {workload.describe()} | {threads} threads | median of {repeat}")
    print()

    header = f"  {'Implementation':<28} {'Tokens':>7} {'Lines':>6} {'Train':>10} {'Infer':>10} {'Total':>10} {'Speedup':>8} {'Accuracy':>9}"
    print(paint(DIM, header))

    for row in rows:
        name = row.name
        tokens = f"{row.tokens:,}" if row.tokens is not None else "-"
        lines = str(row.lines) if row.lines is not None else "-"
        train = f"{row.train_ms:8.1f} ms" if row.train_ms is not None else "     -"
        infer = f"{row.infer_ms:8.1f} ms" if row.infer_ms is not None else "     -"
        total = f"{row.ms:8.1f} ms" if row.ms is not None else "     -"
        if row.ms and baseline_ms and baseline_ms > 0:
            ratio = baseline_ms / row.ms
            speedup = f"{ratio:6.2f}x"
        elif row.ms is None:
            speedup = "     -"
        else:
            speedup = " baseline"
        accuracy = f"{row.accuracy:8.4f}" if row.accuracy is not None else "        -"
        print(
            f"  {name:<28} {tokens:>7} {lines:>6} {train:>10} {infer:>10} {total:>10} {speedup:>8} {accuracy:>9}"
        )
        if row.note:
            print(f"    {paint(DIM, row.note)}")

    print()
    if native_info.get("compiler"):
        print(
            f"  native build  {native_info['compiler']} | {native_info['blas']} | "
            f"compiled in {native_info['compile_ms'] / 1000.0:.1f} s -> "
            f"{Path(native_info['binary']).name}"
        )
    baseline_row = baseline
    if baseline_row and baseline_row.tokens:
        saved = 100.0 * (1.0 - aether_tokens / baseline_row.tokens)
        aether_lines = next((row.lines for row in rows if row.lines is not None), None)
        print(
            f"  token savings {aether_tokens} vs {baseline_row.tokens:,} -> "
            f"{saved:.0f}% fewer tokens; {aether_lines} lines of .ae vs "
            f"{baseline_row.lines} lines of Python"
        )
    if not torch_available:
        print("  pytorch row skipped (torch not installed; set AETHER_TORCH_PATH to enable)")
    print()

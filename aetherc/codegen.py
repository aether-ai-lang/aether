"""Compile validated Aether intent graphs to optimized Python execution code."""

from __future__ import annotations

from .ast_nodes import Arg, Call, Literal, Modifier, Name, Program, Stmt, Task
from .diagnostics import AETHER_VERSION


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


def _value_src(node: object) -> str:
    if isinstance(node, Literal):
        return repr(node.value)
    if isinstance(node, Name):
        return node.id
    return "None"


def _speed(policy: dict | None) -> str:
    if not policy:
        return "balanced"
    value = policy.get("speed", "balanced")
    return value if value in ("max", "balanced", "low") else "balanced"


def _policy_from_modifier(mod: Modifier | None) -> dict | None:
    if mod is None or mod.name != "optimize":
        return None
    policy = {"speed": "balanced"}
    for arg in mod.args:
        if arg.name != "speed":
            continue
        if isinstance(arg.value, Literal) and isinstance(arg.value.value, str):
            policy["speed"] = arg.value.value
        elif isinstance(arg.value, Name):
            policy["speed"] = arg.value.id
    return policy


def _stmt_comment(stmt: Stmt, source_lines: list[str]) -> str:
    text = ""
    if 1 <= stmt.line <= len(source_lines):
        text = source_lines[stmt.line - 1].strip()
    text = text.replace("\t", " ")
    return f"# L{stmt.line}: {text}" if text else f"# L{stmt.line}"


def _expr_src(call: object, task_policy: dict | None) -> str:
    if isinstance(call, Literal):
        return repr(call.value)
    if isinstance(call, Name):
        return call.id
    if not isinstance(call, Call):
        return "None"

    if call.name == "tensor":
        bound = _bind(call, ["rows", "cols"])
        return f"rt.tensor({_value_src(bound.get('rows'))}, {_value_src(bound.get('cols'))})"

    if call.name == "load_data":
        bound = _bind(call, ["samples", "features", "classes"])
        extra = ""
        if "classes" in bound:
            extra = f", classes={_value_src(bound['classes'])}"
        return (
            "rt.load_data("
            f"samples={_value_src(bound.get('samples'))}, "
            f"features={_value_src(bound.get('features'))}"
            f"{extra})"
        )

    if call.name == "classifier":
        bound = _bind(call, ["in", "classes", "hidden"])
        hidden = _value_src(bound["hidden"]) if "hidden" in bound else "128"
        explicit = None
        for mod in call.modifiers:
            if mod.name == "optimize":
                explicit = _policy_from_modifier(mod)
                break
        policy = explicit or task_policy or {"speed": "balanced"}
        return (
            "rt.classifier("
            f"in_dim={_value_src(bound.get('in'))}, "
            f"classes={_value_src(bound.get('classes'))}, "
            f"hidden={hidden}, "
            f"policy={{'speed': '{_speed(policy)}'}})"
        )

    if call.name == "neural_net":
        bound = _bind(call, ["in", "out", "hidden"])
        hidden = _value_src(bound["hidden"]) if "hidden" in bound else "512"
        explicit = None
        for mod in call.modifiers:
            if mod.name == "optimize":
                explicit = _policy_from_modifier(mod)
                break
        policy = explicit or task_policy or {"speed": "balanced"}
        return (
            "rt.neural_net("
            f"input={_value_src(bound.get('in'))}, "
            f"out_dim={_value_src(bound.get('out'))}, "
            f"hidden={hidden}, "
            f"policy={{'speed': '{_speed(policy)}'}})"
        )

    if call.name == "run":
        bound = _bind(call, ["data"])
        target = call.target or "None"
        return f"rt.stream({target}, {_value_src(bound.get('data'))})"

    return f"rt.unsupported({call.name!r})"


def _stmt_src(stmt: Stmt, task_policy: dict | None, source_lines: list[str]) -> list[str]:
    rendered: list[str] = []
    if stmt.kind == "assign":
        rendered.append(f"{stmt.name} = {_expr_src(stmt.expr, task_policy)}")
    elif stmt.kind == "emit":
        rendered.append(f"rt.emit({_expr_src(stmt.expr, task_policy)})")
    elif stmt.kind == "return":
        rendered.append(f"return {_expr_src(stmt.expr, task_policy)}")
    elif stmt.kind == "train":
        rendered.append(f"rt.train({stmt.target}, {stmt.data}, epochs={stmt.epochs})")
    elif stmt.kind == "export":
        rendered.append(f"rt.export({stmt.target}, {stmt.path!r})")
    else:
        rendered.append(_expr_src(stmt.expr, task_policy))

    return [_stmt_comment(stmt, source_lines)] + rendered


def _task_function(task: Task, source_lines: list[str]) -> list[str]:
    policy = _policy_from_modifier(task.modifier)
    lines = [f"    def task_{task.name}(rt):"]
    body: list[str] = []
    for stmt in task.body:
        for source in _stmt_src(stmt, policy, source_lines):
            body.append(f"        {source}")
    if not body:
        body.append("        return None")
    return lines + body


def generate(program: Program, source_name: str) -> str:
    """Return a Python module implementing the compiled task graph."""

    task_names = [task.name for task in program.tasks]
    plan = {
        "aether": AETHER_VERSION,
        "source": source_name,
        "target": "auto_hardware",
        "tasks": task_names,
        "optimizations": [
            "float32",
            "blas-sgemm",
            "fused-matmul-bias-relu",
            "preallocated-chunk-buffers",
            "streaming-inference",
            "sgd-momentum-training",
            "self-healing-guards",
            "aeb1-model-export",
        ],
    }

    out: list[str] = [
        "# ---------------------------------------------------------------------------",
        f"# Aether {AETHER_VERSION} -- generated execution plan. DO NOT EDIT.",
        f"# source: {source_name}",
        "# target: auto_hardware (backend resolved at runtime)",
        "# plan:   f32 | BLAS sgemm | fused matmul+bias+relu | preallocated chunk buffers",
        "# ---------------------------------------------------------------------------",
        "from __future__ import annotations",
        "",
        f"PLAN = {plan!r}",
        "",
        "",
        "def build(runtime):",
        '    """Materialize the compiled Aether task graph."""',
        "    tasks = {}",
        "",
    ]

    for index, task in enumerate(program.tasks):
        out.extend(_task_function(task, program.lines))
        out.append(f"    tasks[{task.name!r}] = task_{task.name}")
        if index != len(program.tasks) - 1:
            out.append("")

    out.extend(["", "    return tasks", ""])
    return "\n".join(out)

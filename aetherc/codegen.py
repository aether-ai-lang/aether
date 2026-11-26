"""Compile validated Aether intent graphs to optimized Python execution code."""

from __future__ import annotations

from .ast_nodes import Arg, Binary, Call, Function, Literal, Modifier, Name, Program, Stmt, Task, Unary
from .diagnostics import AETHER_VERSION

_USER_FNS: set[str] = set()


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
    if isinstance(call, Binary):
        op = call.op
        left = _expr_src(call.left, task_policy)
        right = _expr_src(call.right, task_policy)
        if op == "%":
            return f"({left} % {right})"
        if op in ("==", "!=", ">", "<", ">=", "<="):
            return f"({left} {op} {right})"
        if op == "/":
            return f"({left} / {right})"
        return f"({left} {op} {right})"
    if isinstance(call, Unary):
        return f"(-{_expr_src(call.operand, task_policy)})"
    if isinstance(call, Literal):
        return repr(call.value)
    if isinstance(call, Name):
        return call.id
    if not isinstance(call, Call):
        return "None"

    if call.name in _USER_FNS:
        args = []
        for arg in call.args:
            args.append(_expr_src(arg.value, task_policy))
        return f"ae_fn_{call.name}({', '.join(args)})"

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

    if call.name == "load_csv":
        bound = _bind(call, ["path", "target", "classes"])
        return (
            f"rt.load_csv({_value_src(bound.get('path'))}, "
            f"target={_value_src(bound.get('target'))})"
        )
    if call.name == "save_json":
        bound = _bind(call, ["value", "path"])
        return (
            f"rt.save_json({_value_src(bound.get('value'))}, "
            f"{_value_src(bound.get('path'))})"
        )
    if call.name == "load_model":
        bound = _bind(call, ["path"])
        return f"rt.load_model({_value_src(bound.get('path'))})"

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


def _stmt_src(
    stmt: Stmt,
    task_policy: dict | None,
    source_lines: list[str],
    depth: int = 0,
    base: int = 2,
) -> list[str]:
    """Render one statement; every returned line is fully indented."""
    pad = "    " * (base + depth)
    out: list[str] = [pad + _stmt_comment(stmt, source_lines)]

    def render_inner(body: list[Stmt], extra_depth: int) -> list[str]:
        lines: list[str] = []
        for inner in body:
            lines.extend(
                _stmt_src(inner, task_policy, source_lines, depth + extra_depth, base)
            )
        return lines or [pad + "    pass"]

    if stmt.kind in ("assign", "let"):
        out.append(pad + f"{stmt.name} = {_expr_src(stmt.expr, task_policy)}")
    elif stmt.kind == "repeat":
        out.append(pad + f"for _ae_loop in range(int({_expr_src(stmt.count, task_policy)})):")
        out.extend(render_inner(stmt.body, 1))
    elif stmt.kind == "while":
        max_iters = int(stmt.max_iters or 100_000)
        out.append(pad + f"_ae_guard = 0")
        out.append(pad + f"while {_expr_src(stmt.expr, task_policy)}:")
        guard_body = [
            pad + "    _ae_guard += 1",
            pad + f"    if _ae_guard > {max_iters}:",
            pad + f"        rt.while_guard({max_iters})",
            pad + "        break",
        ]
        out.extend(guard_body)
        out.extend(render_inner(stmt.body, 1))
    elif stmt.kind == "train":
        out.append(
            pad
            + f"accuracy = rt.train({stmt.target}, {stmt.data}, epochs={stmt.epochs}).accuracy"
        )
    elif stmt.kind == "if":
        out.append(pad + f"if {_expr_src(stmt.expr, task_policy)}:")
        out.extend(render_inner(stmt.body, 1))
        out.append(pad + "else:")
        out.extend(render_inner(stmt.else_body, 1))
    elif stmt.kind == "log":
        out.append(pad + f"rt.log({_expr_src(stmt.expr, task_policy)})")
    elif stmt.kind == "emit":
        out.append(pad + f"rt.emit({_expr_src(stmt.expr, task_policy)})")
    elif stmt.kind == "return":
        out.append(pad + f"return {_expr_src(stmt.expr, task_policy)}")
    elif stmt.kind == "train":
        out.append(pad + f"rt.train({stmt.target}, {stmt.data}, epochs={stmt.epochs})")
    elif stmt.kind == "export":
        out.append(pad + f"rt.export({stmt.target}, {stmt.path!r})")
    elif stmt.kind == "serve":
        route = "/predict"
        for mod in stmt.modifiers:
            if mod.name == "route" and mod.args:
                value = mod.args[0].value
                if isinstance(value, Literal) and isinstance(value.value, str):
                    route = value.value
                    break
        out.append(pad + f"rt.serve({stmt.target}, port={stmt.port}, route={route!r})")
    else:
        out.append(pad + _expr_src(stmt.expr, task_policy))

    return out


def _iter_stmts(body: list[Stmt]):
    for stmt in body:
        yield stmt
        if stmt.body:
            yield from _iter_stmts(stmt.body)
        if stmt.else_body:
            yield from _iter_stmts(stmt.else_body)


def _task_function(task: Task, source_lines: list[str]) -> list[str]:
    policy = _policy_from_modifier(task.modifier)
    lines = [f"    def task_{task.name}(rt):"]
    body: list[str] = []
    # `accuracy` is the built-in training binding; tasks that reference it in a
    # loop condition read it before the first train assignment.
    if any(stmt.kind == "train" for stmt in _iter_stmts(task.body)):
        body.append("        accuracy = 0.0")
    for stmt in task.body:
        body.extend(_stmt_src(stmt, policy, source_lines))
    if not body:
        body.append("        return None")
    return lines + body


def _fn_def(fn: Function, program: Program) -> list[str]:
    lines = [f"def ae_fn_{fn.name}({', '.join(fn.params)}):"]
    body: list[str] = []
    for stmt in fn.body:
        body.extend(_stmt_src(stmt, None, program.lines, depth=0, base=1))
    if not body:
        body.append("    pass")
    default = {"float": "0.0", "int": "0", "bool": "False", "str": '""'}[fn.ret_type]
    body.append(f"    return {default}  # fall-through guard")
    return lines + body


def generate(program: Program, source_name: str) -> str:
    """Return a Python module implementing the compiled task graph."""

    _USER_FNS.clear()
    _USER_FNS.update(fn.name for fn in program.functions)

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
    ]

    for fn in program.functions:
        out.extend(_fn_def(fn, program))
        out.append("")

    out.extend(
        [
            "def build(runtime):",
            '    """Materialize the compiled Aether task graph."""',
            "    tasks = {}",
            "",
        ]
    )

    for index, task in enumerate(program.tasks):
        out.extend(_task_function(task, program.lines))
        out.append(f"    tasks[{task.name!r}] = task_{task.name}")
        if index != len(program.tasks) - 1:
            out.append("")

    out.extend(["", "    return tasks", ""])
    return "\n".join(out)

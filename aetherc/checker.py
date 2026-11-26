"""Semantic analysis, intent type-checking, and self-healing diagnostics."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .ast_nodes import Arg, Binary, Call, Literal, Modifier, Name, Program, Stmt, Task, Unary
from .diagnostics import (
    Diagnostic,
    delete_line_patch,
    insert_line_patch,
    replace_line_patch,
    replace_span_patch,
)

ALLOWED_MODULES = {"ai.core", "web.core", "agent.core"}
KNOWN_INTENTS = ("tensor", "load_data", "load_model", "neural_net", "classifier", "run")
SPEED_VALUES = ("max", "balanced", "low")
MAX_DIM = 16384
MAX_ELEMENTS = 100_000_000
MAX_SAMPLES = 1_000_000
MAX_CLASSES = 1_000
MAX_PORT = 65535
MAX_REPEAT = 1_000_000

SIGNATURES = {
    "tensor": "tensor(rows, cols) -> auto_hardware",
    "load_data": "load_data(samples: N, features: D, classes: K?) -> auto_hardware",
    "load_model": 'load_model("model.ae.bin") -> auto_hardware',
    "neural_net": "neural_net(in: X, out: Y, hidden: N?) -> optimize(speed: max|balanced|low)",
    "classifier": "classifier(in: D, classes: K, hidden: N?) -> optimize(speed: max|balanced|low)",
    "run": "run model(data) -> stream_output",
}


@dataclass
class Ty:
    kind: str  # tensor | dataset | model | stream | int | float | str | error
    shape: tuple[int, int] | None = None
    in_dim: int | None = None
    hidden: int | None = None
    out_dim: int | None = None
    policy: dict | None = None
    model_kind: str = "mlp"
    classes: int | None = None
    samples: int | None = None
    features: int | None = None
    trained: bool = False
    decl_line: int = 0
    decl_column: int = 0


ERROR = Ty("error")


class Checker:
    def __init__(self, program: Program) -> None:
        self.program = program
        self.path = program.path
        self.lines = program.lines
        self.diags: list[Diagnostic] = []
        self._seen: set[tuple] = set()
        self.uses: set[str] = {u.module for u in program.uses}
        self.core_avail = "ai.core" in self.uses
        self.web_avail = "web.core" in self.uses
        self.agent_avail = "agent.core" in self.uses
        self.reported_missing_core = False
        self.reported_missing_web = False
        self.reported_missing_agent = False

    # ------------------------------------------------------------------ utils
    def error(
        self,
        code: str,
        message: str,
        line: int,
        column: int,
        hint: str | None = None,
        patch: str | None = None,
        severity: str = "error",
    ) -> None:
        key = (code, line, column, message, severity)
        if key in self._seen:
            return
        self._seen.add(key)
        self.diags.append(
            Diagnostic(
                code=code,
                message=message,
                line=line,
                column=column,
                hint=hint,
                fix_patch=patch,
                severity=severity,
            )
        )

    @staticmethod
    def _suggest(word: str, options: tuple[str, ...] | list[str]) -> str | None:
        matches = difflib.get_close_matches(word, list(options), n=1, cutoff=0.5)
        return matches[0] if matches else None

    def _append_modifier(self, lineno: int, text: str) -> str | None:
        if not 1 <= lineno <= len(self.lines):
            return None
        source = self.lines[lineno - 1]
        if "#" in source:
            return None
        return replace_line_patch(self.path, self.lines, lineno, source.rstrip() + " " + text)

    # ------------------------------------------------------------- top level
    def check(self) -> list[Diagnostic]:
        self.check_uses()
        if not self.program.tasks:
            self.error(
                "AE0501",
                "Program defines no tasks",
                1,
                1,
                hint="Add an entrypoint, e.g. 'task Main:' followed by an indented intent call.",
            )

        seen: set[str] = set()
        for task in self.program.tasks:
            if task.name in seen:
                self.error(
                    "AE0500",
                    f"Duplicate task '{task.name}'",
                    task.line,
                    task.column,
                    hint="Task names must be unique; rename one of them.",
                )
            seen.add(task.name)
            self.check_task(task)

        self.diags.sort(key=lambda d: (d.line, d.column, 0 if d.severity == "error" else 1))
        return self.diags

    def check_uses(self) -> None:
        seen: dict[str, int] = {}
        first_task_line = self.program.tasks[0].line if self.program.tasks else 10**9
        for use in self.program.uses:
            if use.module in seen:
                self.error(
                    "AE0200",
                    f"Duplicate module '{use.module}'",
                    use.line,
                    use.column,
                    hint=f"Remove the repeated 'use {use.module}'.",
                    patch=delete_line_patch(self.path, self.lines, use.line),
                )
                continue
            seen[use.module] = use.line

            if use.module not in ALLOWED_MODULES:
                suggestion = self._suggest(use.module, tuple(sorted(ALLOWED_MODULES)))
                hint = f"v0.1 provides: {', '.join(sorted(ALLOWED_MODULES))}."
                patch = None
                if suggestion:
                    hint = f"Did you mean 'use {suggestion}'?"
                    patch = replace_span_patch(
                        self.path,
                        self.lines,
                        use.line,
                        use.column + 4,
                        len(use.module),
                        suggestion,
                    )
                self.error(
                    "AE0200",
                    f"Unknown module '{use.module}'",
                    use.line,
                    use.column,
                    hint=hint,
                    patch=patch,
                )

            if use.line > first_task_line:
                self.error(
                    "AE0200",
                    "'use' declarations must appear before tasks",
                    use.line,
                    use.column,
                    hint=f"Move 'use {use.module}' above the first task declaration.",
                )

    # ------------------------------------------------------------------ tasks
    def check_task(self, task: Task) -> None:
        default_policy: dict | None = None
        if task.modifier is not None:
            if task.modifier.name != "optimize":
                suggestion = self._suggest(task.modifier.name, ("optimize",))
                patch = None
                if suggestion:
                    patch = replace_span_patch(
                        self.path,
                        self.lines,
                        task.modifier.line,
                        task.modifier.column,
                        len(task.modifier.name),
                        suggestion,
                    )
                self.error(
                    "AE0302",
                    f"Unknown task modifier '-> {task.modifier.name}'",
                    task.modifier.line,
                    task.modifier.column,
                    hint="Only '-> optimize(speed: max|balanced|low)' is valid on a task.",
                    patch=patch,
                )
            else:
                default_policy = self.parse_policy(task.modifier)

        if not task.body:
            self.error(
                "AE0100",
                f"Task '{task.name}' has an empty body",
                task.line,
                task.column,
                hint="Add at least one intent statement, e.g. 'data = tensor(64, 64) -> auto_hardware'.",
            )
            return

        scope: dict[str, Ty] = {}
        for stmt in task.body:
            self.check_stmt(stmt, scope, default_policy)

    def check_stmt(
        self,
        stmt: Stmt,
        scope: dict[str, Ty],
        default_policy: dict | None,
        inside_block: bool = False,
    ) -> None:
        if stmt.kind == "let":
            self.check_let(stmt, scope)
            return

        if stmt.kind == "repeat":
            self.check_repeat(stmt, scope, default_policy)
            return

        if stmt.kind == "if":
            self.check_if(stmt, scope, default_policy)
            return

        if stmt.kind == "log":
            self.check_log(stmt, scope)
            return

        if stmt.kind == "serve" and inside_block:
            self.error(
                "AE0210",
                "'serve' must be a top-level task statement",
                stmt.line,
                stmt.column,
                hint="A program serves exactly once; move 'serve' out of loops and conditionals.",
            )
            return

        if stmt.kind == "train":
            self.check_train(stmt, scope)
            return

        if stmt.kind == "export":
            self.check_export(stmt, scope)
            return

        if stmt.kind == "serve":
            self.check_serve(stmt, scope)
            return

        if stmt.kind == "assign":
            ty = self.check_expr(stmt.expr, scope, default_policy)
            scope[stmt.name or ""] = ty
            return

        if stmt.kind == "emit":
            ty = self.check_expr(stmt.expr, scope, default_policy)
            if ty.kind == "model":
                self.error(
                    "AE0210",
                    "Cannot emit a model; run it first",
                    stmt.line,
                    stmt.column,
                    hint="Use 'run model(data) -> stream_output' and emit the resulting stream.",
                )
            return

        if stmt.kind == "return":
            self.check_expr(stmt.expr, scope, default_policy)
            return

        if not isinstance(stmt.expr, Call):
            self.error(
                "AE0100",
                "Expression statement has no effect",
                stmt.line,
                stmt.column,
                hint="Statements are assignments, 'emit value', or an intent call such as 'run model(data) -> stream_output'.",
            )
            return
        self.check_expr(stmt.expr, scope, default_policy)

    # ------------------------------------------------------------ expressions
    def check_expr(self, expr: object, scope: dict[str, Ty], default_policy: dict | None) -> Ty:
        if isinstance(expr, Binary):
            return self.check_binary(expr, scope, default_policy)

        if isinstance(expr, Unary):
            operand = self.check_expr(expr.operand, scope, default_policy)
            if operand.kind not in ("int", "float"):
                self.error(
                    "AE0210",
                    "unary '-' requires a number",
                    expr.line,
                    expr.column,
                    hint="Example: let delta = -0.5",
                )
                return ERROR
            return operand

        if isinstance(expr, Literal):
            if isinstance(expr.value, bool):
                return Ty("int")
            if isinstance(expr.value, int):
                return Ty("int")
            if isinstance(expr.value, float):
                return Ty("float")
            if isinstance(expr.value, str):
                return Ty("str")
            return Ty("str")

        if isinstance(expr, Name):
            return self.lookup(expr, scope)

        if isinstance(expr, Call):
            return self.check_call(expr, scope, default_policy)

        self.error("AE0100", "Invalid expression", 1, 1)
        return ERROR

    def check_binary(self, expr: Binary, scope: dict[str, Ty], default_policy: dict | None) -> Ty:
        left = self.check_expr(expr.left, scope, default_policy)
        right = self.check_expr(expr.right, scope, default_policy)

        if expr.op in ("==", "!=", ">", "<", ">=", "<="):
            numeric = ("int", "float")
            comparable = (
                left.kind in numeric and right.kind in numeric
            ) or (left.kind == "str" and right.kind == "str" and expr.op in ("==", "!="))
            if not comparable:
                self.error(
                    "AE0210",
                    f"cannot compare {self.article(left.kind)} with {self.article(right.kind)}",
                    expr.line,
                    expr.column,
                    hint="Comparisons work on numbers (== != also on strings).",
                )
                return ERROR
            return Ty("bool")

        if left.kind not in ("int", "float") or right.kind not in ("int", "float"):
            self.error(
                "AE0210",
                f"arithmetic requires numbers, got {left.kind} {expr.op} {right.kind}",
                expr.line,
                expr.column,
                hint="Arithmetic works on int/float scalars; tensors and models are not numbers.",
            )
            return ERROR
        if expr.op == "%" and (left.kind != "int" or right.kind != "int"):
            self.error(
                "AE0210",
                "'%' requires two integers",
                expr.line,
                expr.column,
                hint="Example: let phase = count % 2",
            )
            return ERROR
        if expr.op == "/" or left.kind == "float" or right.kind == "float":
            return Ty("float")
        return Ty("int")

    def lookup(self, name: Name, scope: dict[str, Ty]) -> Ty:
        if name.id in scope:
            return scope[name.id]

        suggestion = self._suggest(name.id, list(scope))
        hint = "Define it earlier in the task before use."
        patch = None
        if suggestion:
            hint = f"Did you mean '{suggestion}'?"
            patch = replace_span_patch(
                self.path, self.lines, name.line, name.column, len(name.id), suggestion
            )
        self.error("AE0211", f"Undefined symbol '{name.id}'", name.line, name.column, hint=hint, patch=patch)
        return ERROR

    def require_core(self, call: Call) -> None:
        if self.core_avail or self.reported_missing_core:
            return
        self.reported_missing_core = True

        if self.program.uses:
            insert_at = max(u.line for u in self.program.uses) + 1
        elif self.program.tasks:
            insert_at = self.program.tasks[0].line
        else:
            insert_at = 1
        self.error(
            "AE0200",
            "Missing 'use ai.core' for AI intents",
            call.line,
            call.column,
            hint="Add 'use ai.core' above the first task.",
            patch=insert_line_patch(self.path, self.lines, insert_at, "use ai.core"),
        )

    def require_web_core(self, stmt: Stmt) -> None:
        if self.web_avail or self.reported_missing_web:
            return
        self.reported_missing_web = True

        if self.program.uses:
            insert_at = max(u.line for u in self.program.uses) + 1
        elif self.program.tasks:
            insert_at = self.program.tasks[0].line
        else:
            insert_at = 1
        self.error(
            "AE0200",
            "Missing 'use web.core' for web intents",
            stmt.line,
            stmt.column,
            hint="Add 'use web.core' above the first task to serve models over HTTP.",
            patch=insert_line_patch(self.path, self.lines, insert_at, "use web.core"),
        )

    def require_agent_core(self, stmt: Stmt) -> None:
        if self.agent_avail or self.reported_missing_agent:
            return
        self.reported_missing_agent = True

        if self.program.uses:
            insert_at = max(u.line for u in self.program.uses) + 1
        elif self.program.tasks:
            insert_at = self.program.tasks[0].line
        else:
            insert_at = 1
        self.error(
            "AE0200",
            "Missing 'use agent.core' for agent loops",
            stmt.line,
            stmt.column,
            hint="Add 'use agent.core' above the first task to use 'repeat ... times:'.",
            patch=insert_line_patch(self.path, self.lines, insert_at, "use agent.core"),
        )

    def check_call(self, call: Call, scope: dict[str, Ty], default_policy: dict | None) -> Ty:
        if call.name not in KNOWN_INTENTS:
            suggestion = self._suggest(call.name, KNOWN_INTENTS)
            patch = None
            hint = f"v0.3 intents: {', '.join(KNOWN_INTENTS)}."
            if suggestion:
                hint = f"Did you mean '{suggestion}'?"
                patch = replace_span_patch(
                    self.path, self.lines, call.line, call.column, len(call.name), suggestion
                )
            self.error(
                "AE0201",
                f"Unknown intent '{call.name}'",
                call.line,
                call.column,
                hint=hint,
                patch=patch,
            )
            return ERROR

        self.require_core(call)

        if call.name == "tensor":
            return self.check_tensor(call)
        if call.name == "load_data":
            return self.check_load_data(call)
        if call.name == "load_model":
            return self.check_load_model(call)
        if call.name == "neural_net":
            return self.check_neural_net(call, scope, default_policy)
        if call.name == "classifier":
            return self.check_classifier(call, default_policy)
        return self.check_run(call, scope)

    # ------------------------------------------------------------ intent: tensor
    def check_tensor(self, call: Call) -> Ty:
        bound, _ = self.bind(call, [("rows", True), ("cols", True)])

        modifier_names = [m.name for m in call.modifiers]
        if "auto_hardware" not in modifier_names:
            self.error(
                "AE0300",
                "tensor(...) requires '-> auto_hardware'",
                call.line,
                call.column,
                hint="Append '-> auto_hardware' so the compiler can bind a device.",
                patch=self._append_modifier(call.line, "-> auto_hardware"),
            )

        seen_modifiers: set[str] = set()
        for mod in call.modifiers:
            if mod.name != "auto_hardware":
                suggestion = self._suggest(mod.name, ("auto_hardware",))
                patch = None
                if suggestion:
                    patch = replace_span_patch(
                        self.path, self.lines, mod.line, mod.column, len(mod.name), suggestion
                    )
                self.error(
                    "AE0302",
                    f"Unknown modifier '-> {mod.name}'",
                    mod.line,
                    mod.column,
                    hint="Only '-> auto_hardware' is valid for tensor().",
                    patch=patch,
                )
                continue
            if mod.name in seen_modifiers:
                self.error(
                    "AE0302",
                    "Duplicate '-> auto_hardware' modifier",
                    mod.line,
                    mod.column,
                    hint="Remove the repeated modifier.",
                )
            seen_modifiers.add(mod.name)
            if mod.args:
                self.error(
                    "AE0212",
                    "auto_hardware takes no arguments",
                    mod.args[0].line,
                    mod.args[0].column,
                    hint="Write '-> auto_hardware' without parentheses.",
                )

        rows = self.int_arg(bound.get("rows"), "rows")
        cols = self.int_arg(bound.get("cols"), "cols")
        if rows is None or cols is None:
            return ERROR

        valid = True
        for label, value in (("rows", rows), ("cols", cols)):
            if not 1 <= value <= MAX_DIM:
                self.error(
                    "AE0212",
                    f"{label} must be between 1 and {MAX_DIM}",
                    call.line,
                    call.column,
                    hint=f"Use a smaller dimension, e.g. {label}: 1024.",
                )
                valid = False
        if rows * cols > MAX_ELEMENTS:
            self.error(
                "AE0212",
                f"tensor is too large ({rows * cols:,} elements)",
                call.line,
                call.column,
                hint=f"Keep rows*cols <= {MAX_ELEMENTS:,} in v0.1.",
            )
            valid = False

        return Ty("tensor", shape=(rows, cols)) if valid else ERROR

    # --------------------------------------------------------- intent: load_data
    def check_load_data(self, call: Call) -> Ty:
        bound, _ = self.bind(
            call, [("samples", True), ("features", True), ("classes", False)]
        )

        modifier_names = [m.name for m in call.modifiers]
        if "auto_hardware" not in modifier_names:
            self.error(
                "AE0300",
                "load_data(...) requires '-> auto_hardware'",
                call.line,
                call.column,
                hint="Append '-> auto_hardware' so the compiler can bind a device.",
                patch=self._append_modifier(call.line, "-> auto_hardware"),
            )

        seen_modifiers: set[str] = set()
        for mod in call.modifiers:
            if mod.name != "auto_hardware":
                suggestion = self._suggest(mod.name, ("auto_hardware",))
                patch = None
                if suggestion:
                    patch = replace_span_patch(
                        self.path, self.lines, mod.line, mod.column, len(mod.name), suggestion
                    )
                self.error(
                    "AE0302",
                    f"Unknown modifier '-> {mod.name}'",
                    mod.line,
                    mod.column,
                    hint="Only '-> auto_hardware' is valid for load_data().",
                    patch=patch,
                )
                continue
            if mod.name in seen_modifiers:
                self.error(
                    "AE0302",
                    "Duplicate '-> auto_hardware' modifier",
                    mod.line,
                    mod.column,
                    hint="Remove the repeated modifier.",
                )
            seen_modifiers.add(mod.name)
            if mod.args:
                self.error(
                    "AE0212",
                    "auto_hardware takes no arguments",
                    mod.args[0].line,
                    mod.args[0].column,
                    hint="Write '-> auto_hardware' without parentheses.",
                )

        samples = self.int_arg(bound.get("samples"), "samples")
        features = self.int_arg(bound.get("features"), "features")
        classes = 10
        if "classes" in bound:
            parsed_classes = self.int_arg(bound.get("classes"), "classes")
            if parsed_classes is None:
                return ERROR
            classes = parsed_classes

        if samples is None or features is None:
            return ERROR

        valid = True
        if not 1 <= samples <= MAX_SAMPLES:
            self.error(
                "AE0212",
                f"samples must be between 1 and {MAX_SAMPLES:,}",
                call.line,
                call.column,
                hint="Example: load_data(samples: 10000, features: 128)",
            )
            valid = False
        if not 1 <= features <= 4096:
            self.error(
                "AE0212",
                "features must be between 1 and 4096",
                call.line,
                call.column,
                hint="Example: load_data(samples: 10000, features: 128)",
            )
            valid = False
        if not 2 <= classes <= MAX_CLASSES:
            self.error(
                "AE0212",
                f"classes must be between 2 and {MAX_CLASSES}",
                call.line,
                call.column,
                hint="The default is classes: 10.",
            )
            valid = False
        if samples * features > MAX_ELEMENTS:
            self.error(
                "AE0212",
                f"dataset is too large ({samples * features:,} elements)",
                call.line,
                call.column,
                hint=f"Keep samples*features <= {MAX_ELEMENTS:,} in v0.2.",
            )
            valid = False

        if not valid:
            return ERROR
        return Ty(
            "dataset",
            shape=(samples, features),
            samples=samples,
            features=features,
            classes=classes,
        )

    # -------------------------------------------------------- intent: load_model
    def check_load_model(self, call: Call) -> Ty:
        bound, _ = self.bind(call, [("path", True)])
        node = bound.get("path")

        path: str | None = None
        if isinstance(node, Literal) and isinstance(node.value, str):
            path = node.value
        elif node is not None:
            self.error(
                "AE0212",
                "load_model requires a quoted path",
                getattr(node, "line", call.line),
                getattr(node, "column", call.column),
                hint='Example: load_model("model.ae.bin")',
            )

        if "auto_hardware" not in [m.name for m in call.modifiers]:
            self.error(
                "AE0300",
                "load_model(...) requires '-> auto_hardware'",
                call.line,
                call.column,
                hint="Append '-> auto_hardware' so the compiler can bind a device.",
                patch=self._append_modifier(call.line, "-> auto_hardware"),
            )

        seen: set[str] = set()
        for mod in call.modifiers:
            if mod.name != "auto_hardware":
                suggestion = self._suggest(mod.name, ("auto_hardware",))
                patch = None
                if suggestion:
                    patch = replace_span_patch(
                        self.path, self.lines, mod.line, mod.column, len(mod.name), suggestion
                    )
                self.error(
                    "AE0302",
                    f"Unknown modifier '-> {mod.name}'",
                    mod.line,
                    mod.column,
                    hint="Only '-> auto_hardware' is valid for load_model().",
                    patch=patch,
                )
                continue
            if mod.name in seen:
                self.error(
                    "AE0302",
                    "Duplicate '-> auto_hardware' modifier",
                    mod.line,
                    mod.column,
                    hint="Remove the repeated modifier.",
                )
            seen.add(mod.name)
            if mod.args:
                self.error(
                    "AE0212",
                    "auto_hardware takes no arguments",
                    mod.args[0].line,
                    mod.args[0].column,
                    hint="Write '-> auto_hardware' without parentheses.",
                )

        if path is None:
            return ERROR
        if not path:
            self.error(
                "AE0212",
                "load_model path must not be empty",
                call.line,
                call.column,
                hint='Example: load_model("model.ae.bin")',
            )
            return ERROR
        if len(path) > 240:
            self.error(
                "AE0212",
                "load_model path is too long",
                call.line,
                call.column,
                hint="Keep the path under 240 characters.",
            )
            return ERROR

        return Ty(
            "model",
            model_kind="served",
            trained=True,
            decl_line=call.line,
            decl_column=call.column,
        )

    # --------------------------------------------------------- intent: classifier
    def check_classifier(self, call: Call, default_policy: dict | None) -> Ty:
        bound, _ = self.bind(
            call, [("in", True), ("classes", True), ("hidden", False)]
        )

        in_dim = self.int_arg(bound.get("in"), "in")
        classes = self.int_arg(bound.get("classes"), "classes")
        hidden = 128
        if "hidden" in bound:
            parsed_hidden = self.int_arg(bound.get("hidden"), "hidden")
            if parsed_hidden is None:
                return ERROR
            hidden = parsed_hidden

        if in_dim is None or classes is None:
            return ERROR

        valid = True
        if not 1 <= in_dim <= MAX_DIM:
            self.error(
                "AE0212",
                f"in must be between 1 and {MAX_DIM}",
                call.line,
                call.column,
                hint="Example: classifier(in: 128, classes: 10)",
            )
            valid = False
        if not 2 <= classes <= MAX_CLASSES:
            self.error(
                "AE0212",
                f"classes must be between 2 and {MAX_CLASSES}",
                call.line,
                call.column,
                hint="Example: classifier(in: 128, classes: 10)",
            )
            valid = False
        if not 16 <= hidden <= 8192:
            self.error(
                "AE0212",
                "hidden must be between 16 and 8192",
                call.line,
                call.column,
                hint="Example: hidden: 128",
            )
            valid = False

        policy = dict(default_policy or {"speed": "balanced"})
        optimize_mods = [m for m in call.modifiers if m.name == "optimize"]
        other_mods = [m for m in call.modifiers if m.name != "optimize"]

        for mod in other_mods:
            suggestion = self._suggest(mod.name, ("optimize",))
            patch = None
            if suggestion:
                patch = replace_span_patch(
                    self.path, self.lines, mod.line, mod.column, len(mod.name), suggestion
                )
            self.error(
                "AE0302",
                f"Unknown modifier '-> {mod.name}'",
                mod.line,
                mod.column,
                hint="classifier accepts '-> optimize(speed: max|balanced|low)'.",
                patch=patch,
            )

        if len(optimize_mods) > 1:
            self.error(
                "AE0302",
                "Duplicate '-> optimize(...)' modifier",
                optimize_mods[1].line,
                optimize_mods[1].column,
                hint="Keep a single optimize policy.",
            )

        if not optimize_mods:
            self.error(
                "AE0401",
                "classifier has no optimization policy",
                call.line,
                call.column,
                severity="warning",
                hint="Add '-> optimize(speed: balanced)' (or max) for an explicit codegen policy.",
                patch=self._append_modifier(call.line, "-> optimize(speed: balanced)"),
            )
        else:
            explicit = self.parse_policy(optimize_mods[0])
            if explicit is not None:
                policy = explicit

        if not valid:
            return ERROR
        return Ty(
            "model",
            in_dim=in_dim,
            hidden=hidden,
            out_dim=classes,
            policy=policy,
            model_kind="classifier",
            classes=classes,
            trained=False,
            decl_line=call.line,
            decl_column=call.column,
        )

    # --------------------------------------------------------- statement: train
    def check_train(self, stmt: Stmt, scope: dict[str, Ty]) -> None:
        model_ty: Ty | None = None
        dataset_ty: Ty | None = None

        if stmt.target is None or stmt.target not in scope:
            self.error(
                "AE0211",
                f"Undefined model '{stmt.target}'",
                stmt.line,
                stmt.column,
                hint="Declare it first: model = classifier(in: 128, classes: 10)",
            )
        else:
            model_ty = scope[stmt.target]
            if model_ty.kind != "model":
                self.error(
                    "AE0210",
                    f"'{stmt.target}' is {self.article(model_ty.kind)}, not a model",
                    stmt.line,
                    stmt.column,
                    hint="Only classifier(...) models can be trained.",
                )
                model_ty = None
            elif model_ty.model_kind != "classifier":
                self.error(
                    "AE0210",
                    f"'{stmt.target}' is not trainable",
                    stmt.line,
                    stmt.column,
                    hint="Use classifier(in: D, classes: K) for trainable models.",
                )
                model_ty = None

        if stmt.data is None or stmt.data not in scope:
            self.error(
                "AE0211",
                f"Undefined dataset '{stmt.data}'",
                stmt.line,
                stmt.column,
                hint="Create it first: dataset = load_data(samples: 10000, features: 128) -> auto_hardware",
            )
        else:
            dataset_ty = scope[stmt.data]
            if dataset_ty.kind != "dataset":
                self.error(
                    "AE0210",
                    f"'{stmt.data}' is {self.article(dataset_ty.kind)}, not a dataset",
                    stmt.line,
                    stmt.column,
                    hint="Datasets come from load_data(...).",
                )
                dataset_ty = None

        epochs = stmt.epochs or 0
        if not 1 <= epochs <= 1000:
            self.error(
                "AE0212",
                "epochs must be between 1 and 1000",
                stmt.line,
                stmt.column,
                hint="Example: train model on dataset(epochs: 5) -> self_heal",
            )

        modifier_names = [m.name for m in stmt.modifiers]
        if "self_heal" not in modifier_names:
            self.error(
                "AE0300",
                "train (...) requires '-> self_heal'",
                stmt.line,
                stmt.column,
                hint="Append '-> self_heal' to enable numerical guards and recovery.",
                patch=self._append_modifier(stmt.line, "-> self_heal"),
            )

        seen_heal = False
        for mod in stmt.modifiers:
            if mod.name != "self_heal":
                suggestion = self._suggest(mod.name, ("self_heal",))
                patch = None
                if suggestion:
                    patch = replace_span_patch(
                        self.path, self.lines, mod.line, mod.column, len(mod.name), suggestion
                    )
                self.error(
                    "AE0302",
                    f"Unknown modifier '-> {mod.name}'",
                    mod.line,
                    mod.column,
                    hint="train accepts only '-> self_heal'.",
                    patch=patch,
                )
                continue
            if seen_heal:
                self.error(
                    "AE0302",
                    "Duplicate '-> self_heal' modifier",
                    mod.line,
                    mod.column,
                    hint="Remove the repeated modifier.",
                )
            seen_heal = True
            if mod.args:
                self.error(
                    "AE0212",
                    "self_heal takes no arguments",
                    mod.args[0].line,
                    mod.args[0].column,
                    hint="Write '-> self_heal' without parentheses.",
                )

        if model_ty is not None and dataset_ty is not None:
            if model_ty.in_dim != dataset_ty.features:
                self.error(
                    "AE0210",
                    f"Feature mismatch: classifier expects {model_ty.in_dim}, dataset has {dataset_ty.features}",
                    stmt.line,
                    stmt.column,
                    hint=f"Use classifier(in: {dataset_ty.features}, classes: {dataset_ty.classes}).",
                    patch=self._model_arg_patch(model_ty, "in", dataset_ty.features),
                )
            elif model_ty.classes != dataset_ty.classes:
                self.error(
                    "AE0210",
                    f"Class mismatch: classifier has {model_ty.classes}, dataset has {dataset_ty.classes}",
                    stmt.line,
                    stmt.column,
                    hint=f"Use classifier(in: {dataset_ty.features}, classes: {dataset_ty.classes}).",
                    patch=self._model_arg_patch(model_ty, "classes", dataset_ty.classes),
                )
            else:
                model_ty.trained = True

    # --------------------------------------------------------- statement: serve
    def check_serve(self, stmt: Stmt, scope: dict[str, Ty]) -> None:
        self.require_web_core(stmt)

        if stmt.target is None or stmt.target not in scope:
            self.error(
                "AE0211",
                f"Undefined model '{stmt.target}'",
                stmt.line,
                stmt.column,
                hint='Load or train one first: model = load_model("model.ae.bin") -> auto_hardware',
            )
        else:
            ty = scope[stmt.target]
            if ty.kind != "model":
                self.error(
                    "AE0210",
                    f"'serve' expects a model, but '{stmt.target}' is {self.article(ty.kind)}",
                    stmt.line,
                    stmt.column,
                    hint="Serve the result of load_model(...), classifier(...), or neural_net(...).",
                )

        port = stmt.port or 0
        if not 1 <= port <= MAX_PORT:
            self.error(
                "AE0212",
                f"port must be between 1 and {MAX_PORT}",
                stmt.line,
                stmt.column,
                hint="Example: serve model on port(8080) -> route(\"/predict\")",
            )

        routes = [m for m in stmt.modifiers if m.name == "route"]
        for mod in stmt.modifiers:
            if mod.name != "route":
                suggestion = self._suggest(mod.name, ("route",))
                patch = None
                if suggestion:
                    patch = replace_span_patch(
                        self.path, self.lines, mod.line, mod.column, len(mod.name), suggestion
                    )
                self.error(
                    "AE0302",
                    f"Unknown modifier '-> {mod.name}'",
                    mod.line,
                    mod.column,
                    hint='serve accepts only \'-> route("/predict")\'.',
                    patch=patch,
                )
                continue

        if len(routes) > 1:
            self.error(
                "AE0302",
                "Duplicate '-> route(...)' modifier",
                routes[1].line,
                routes[1].column,
                hint="Keep a single route.",
            )

        if not routes:
            self.error(
                "AE0300",
                "serve requires '-> route(\"/...\")'",
                stmt.line,
                stmt.column,
                hint='Add the HTTP endpoint, e.g. -> route("/predict").',
                patch=self._append_modifier(stmt.line, '-> route("/predict")'),
            )
        else:
            route_mod = routes[0]
            if len(route_mod.args) != 1 or not isinstance(
                route_mod.args[0].value, Literal
            ) or not isinstance(route_mod.args[0].value.value, str):
                self.error(
                    "AE0212",
                    "route requires one quoted path",
                    route_mod.line,
                    route_mod.column,
                    hint='Example: -> route("/predict")',
                )
            else:
                route = route_mod.args[0].value.value
                if not route.startswith("/"):
                    self.error(
                        "AE0212",
                        "route must start with '/'",
                        route_mod.args[0].line,
                        route_mod.args[0].column,
                        hint='Example: -> route("/predict")',
                    )
                elif len(route) > 128:
                    self.error(
                        "AE0212",
                        "route is too long",
                        route_mod.args[0].line,
                        route_mod.args[0].column,
                        hint="Keep the route under 128 characters.",
                    )

    # ------------------------------------------------- statements: v0.4 control flow
    def check_let(self, stmt: Stmt, scope: dict[str, Ty]) -> None:
        if not isinstance(stmt.expr, (Literal, Binary, Unary, Name)):
            self.error(
                "AE0210",
                f"'let {stmt.name}' binds a scalar; use plain assignment for intent results",
                stmt.line,
                stmt.column,
                hint="Example: let threshold = 0.85   |   data = tensor(64, 64) -> auto_hardware",
            )
            return

        ty = self.check_expr(stmt.expr, scope, {})
        if ty.kind not in ("int", "float", "str", "bool"):
            self.error(
                "AE0210",
                f"'let {stmt.name}' must bind an int, float, string, or bool",
                stmt.line,
                stmt.column,
                hint="Tensors, datasets, and models bind with plain assignment (no 'let').",
            )
            return

        previous = scope.get(stmt.name or "")
        if previous is not None and previous.kind != ty.kind and previous.kind != "error":
            self.error(
                "AE0210",
                f"'{stmt.name}' was {self.article(previous.kind)} and cannot become {self.article(ty.kind)}",
                stmt.line,
                stmt.column,
                hint=f"Keep '{stmt.name}' a {previous.kind}, or use a new name.",
            )
            return
        scope[stmt.name or ""] = ty

    def check_repeat(self, stmt: Stmt, scope: dict[str, Ty], default_policy: dict | None) -> None:
        self.require_agent_core(stmt)

        count = stmt.count
        if isinstance(count, Literal):
            if not isinstance(count.value, int):
                self.error(
                    "AE0212",
                    "repeat count must be an integer",
                    count.line,
                    count.column,
                    hint="Example: repeat 10 times:",
                )
            elif not 1 <= count.value <= MAX_REPEAT:
                self.error(
                    "AE0212",
                    f"repeat count must be between 1 and {MAX_REPEAT:,}",
                    count.line,
                    count.column,
                    hint="Split very long runs into stages.",
                )
        elif isinstance(count, Name):
            ty = self.lookup(count, scope)
            if ty.kind != "int":
                self.error(
                    "AE0210",
                    f"repeat count '{count.id}' must be an int",
                    count.line,
                    count.column,
                    hint="Example: let epochs = 5 ... repeat epochs times:",
                )
        else:
            self.error(
                "AE0212",
                "repeat count must be an integer literal or int variable",
                stmt.line,
                stmt.column,
                hint="Example: repeat 10 times:",
            )

        for inner in stmt.body:
            self.check_stmt(inner, scope, default_policy, inside_block=True)

    def check_if(self, stmt: Stmt, scope: dict[str, Ty], default_policy: dict | None) -> None:
        condition = self.check_expr(stmt.expr, scope, default_policy) if stmt.expr else ERROR
        if condition.kind != "bool":
            self.error(
                "AE0210",
                "if requires a comparison, e.g. confidence > threshold",
                stmt.line,
                stmt.column,
                hint="Example: if accuracy > 0.9:  ...  else:  ...",
            )

        for inner in stmt.body:
            self.check_stmt(inner, scope, default_policy, inside_block=True)
        for inner in stmt.else_body:
            self.check_stmt(inner, scope, default_policy, inside_block=True)

    def check_log(self, stmt: Stmt, scope: dict[str, Ty]) -> None:
        ty = self.check_expr(stmt.expr, scope, {})
        if ty.kind not in ("int", "float", "str", "bool"):
            self.error(
                "AE0210",
                f"cannot log {self.article(ty.kind)}",
                stmt.line,
                stmt.column,
                hint="log takes strings and numbers: log \"done\" or log total.",
            )

    # -------------------------------------------------------- statement: export
    def check_export(self, stmt: Stmt, scope: dict[str, Ty]) -> None:
        if stmt.target is None or stmt.target not in scope:
            self.error(
                "AE0211",
                f"Undefined model '{stmt.target}'",
                stmt.line,
                stmt.column,
                hint="Train or declare a model before exporting it.",
            )
            return

        ty = scope[stmt.target]
        if ty.kind != "model":
            self.error(
                "AE0210",
                f"'{stmt.target}' is {self.article(ty.kind)}, not a model",
                stmt.line,
                stmt.column,
                hint="Only models can be exported.",
            )
            return

        if ty.model_kind == "classifier" and not ty.trained:
            self.error(
                "AE0210",
                f"Model '{stmt.target}' must be trained before export",
                stmt.line,
                stmt.column,
                hint="Add 'train model on dataset(epochs: 5) -> self_heal' before exporting.",
            )

        if not stmt.path:
            self.error(
                "AE0212",
                "Export path must be a non-empty string",
                stmt.line,
                stmt.column,
                hint='Example: export model -> "model.ae.bin"',
            )
        elif len(stmt.path) > 240:
            self.error(
                "AE0212",
                "Export path is too long",
                stmt.line,
                stmt.column,
                hint="Keep the path under 240 characters.",
            )

    def _model_arg_patch(self, model_ty: Ty, key: str, new_value: object) -> str | None:
        """A fix_patch rewriting `<key>: N` inside the model's declaration."""
        line_no = model_ty.decl_line
        if not 1 <= line_no <= len(self.lines):
            return None
        text = self.lines[line_no - 1]
        match = re.search(rf"\b{re.escape(key)}\s*:\s*\d+", text)
        if not match:
            return None
        return replace_span_patch(
            self.path, self.lines, line_no, match.start() + 1, len(match.group()), f"{key}: {new_value}"
        )

    # ------------------------------------------------------- intent: neural_net
    def check_neural_net(
        self, call: Call, scope: dict[str, Ty], default_policy: dict | None
    ) -> Ty:
        bound, _ = self.bind(call, [("in", True), ("out", True), ("hidden", False)])

        input_ty: Ty | None = None
        in_node = bound.get("in")
        if in_node is not None:
            input_ty = self.as_tensor(in_node, scope, "in")
            if input_ty is None:
                return ERROR

        out_dim = self.int_arg(bound.get("out"), "out")
        if out_dim is not None and not 1 <= out_dim <= MAX_DIM:
            self.error(
                "AE0212",
                f"out must be between 1 and {MAX_DIM}",
                call.line,
                call.column,
                hint="Choose a smaller output width, e.g. out: 256.",
            )
            out_dim = None

        hidden = 512
        hidden_node = bound.get("hidden")
        if hidden_node is not None:
            parsed = self.int_arg(hidden_node, "hidden")
            if parsed is None:
                return ERROR
            if not 16 <= parsed <= 8192:
                self.error(
                    "AE0212",
                    "hidden must be between 16 and 8192",
                    hidden_node.line,
                    hidden_node.column,
                    hint="Example: hidden: 512",
                )
                return ERROR
            hidden = parsed

        policy = dict(default_policy or {"speed": "balanced"})
        optimize_mods = [m for m in call.modifiers if m.name == "optimize"]
        other_mods = [m for m in call.modifiers if m.name != "optimize"]

        for mod in other_mods:
            suggestion = self._suggest(mod.name, ("optimize",))
            patch = None
            if suggestion:
                patch = replace_span_patch(
                    self.path, self.lines, mod.line, mod.column, len(mod.name), suggestion
                )
            self.error(
                "AE0302",
                f"Unknown modifier '-> {mod.name}'",
                mod.line,
                mod.column,
                hint="neural_net accepts '-> optimize(speed: max|balanced|low)'.",
                patch=patch,
            )

        if len(optimize_mods) > 1:
            self.error(
                "AE0302",
                "Duplicate '-> optimize(...)' modifier",
                optimize_mods[1].line,
                optimize_mods[1].column,
                hint="Keep a single optimize policy.",
            )

        if not optimize_mods:
            self.error(
                "AE0401",
                "neural_net has no optimization policy",
                call.line,
                call.column,
                severity="warning",
                hint="Add '-> optimize(speed: balanced)' (or max) for an explicit codegen policy.",
                patch=self._append_modifier(call.line, "-> optimize(speed: balanced)"),
            )
        else:
            explicit = self.parse_policy(optimize_mods[0])
            if explicit is not None:
                policy = explicit

        if input_ty is None or out_dim is None:
            return ERROR

        return Ty(
            "model",
            in_dim=input_ty.shape[1] if input_ty.shape else None,
            hidden=hidden,
            out_dim=out_dim,
            policy=policy,
            model_kind="mlp",
            trained=True,
            decl_line=call.line,
            decl_column=call.column,
        )

    # ---------------------------------------------------------------- intent: run
    def check_run(self, call: Call, scope: dict[str, Ty]) -> Ty:
        model_ty: Ty | None = None
        if call.target is None:
            self.error(
                "AE0212",
                "run requires a model: 'run model(data)'",
                call.line,
                call.column,
                hint="Declare a model first, e.g. net = neural_net(in: x, out: 256).",
            )
        else:
            model_ty = scope.get(call.target)
            if model_ty is None:
                suggestion = self._suggest(call.target, list(scope))
                hint = f"Define '{call.target}' before running it."
                patch = None
                if suggestion:
                    hint = f"Did you mean '{suggestion}'?"
                    patch = replace_span_patch(
                        self.path,
                        self.lines,
                        call.line,
                        call.column,
                        len("run") + 1 + len(call.target),
                        f"run {suggestion}",
                    )
                self.error(
                    "AE0211",
                    f"Undefined model '{call.target}'",
                    call.line,
                    call.column,
                    hint=hint,
                    patch=patch,
                )
            elif model_ty.kind != "model":
                self.error(
                    "AE0210",
                    f"'run' expects a model, but '{call.target}' is {self.article(model_ty.kind)}",
                    call.line,
                    call.column,
                    hint="Only the result of neural_net(...) or classifier(...) can be run.",
                )
                model_ty = None
            elif model_ty.model_kind == "served":
                self.error(
                    "AE0210",
                    f"'{call.target}' was loaded for serving and has no compile-time shape",
                    call.line,
                    call.column,
                    hint='Expose it over HTTP instead: serve model on port(8080) -> route("/predict").',
                )
                model_ty = None

        bound, _ = self.bind(call, [("data", True)])
        data_ty: Ty | None = None
        data_node = bound.get("data")
        if data_node is not None:
            if model_ty is not None and model_ty.model_kind == "classifier":
                if isinstance(data_node, Name):
                    data_ty = self.lookup(data_node, scope)
                    if data_ty.kind == "error":
                        return ERROR
                    if data_ty.kind != "dataset":
                        self.error(
                            "AE0210",
                            f"'{call.target}' expects a dataset, but 'data' is {self.article(data_ty.kind)}",
                            data_node.line,
                            data_node.column,
                            hint="Classifiers run on datasets created by load_data(...).",
                        )
                        return ERROR
                else:
                    self.error(
                        "AE0210",
                        "'data' must be a dataset",
                        getattr(data_node, "line", call.line),
                        getattr(data_node, "column", call.column),
                        hint="Pass the dataset variable created by load_data(...).",
                    )
                    return ERROR
            else:
                data_ty = self.as_tensor(data_node, scope, "data")
                if data_ty is None:
                    return ERROR

        if model_ty is not None and data_ty is not None:
            width = (
                data_ty.features
                if data_ty.kind == "dataset"
                else (data_ty.shape[1] if data_ty.shape else None)
            )
            if model_ty.in_dim is not None and width is not None and width != model_ty.in_dim:
                self.error(
                    "AE0210",
                    f"Input width mismatch: network expects {model_ty.in_dim}, data has {width}",
                    data_node.line,
                    data_node.column,
                    hint=(
                        f"Allocate data as tensor(rows, {model_ty.in_dim}) "
                        "or rebuild the network with the matching 'in' tensor."
                    ),
                )
                return ERROR

        modifier_names = [m.name for m in call.modifiers]
        if "stream_output" not in modifier_names:
            self.error(
                "AE0301",
                "run (...) requires '-> stream_output'",
                call.line,
                call.column,
                hint="Append '-> stream_output' to emit results chunk by chunk.",
                patch=self._append_modifier(call.line, "-> stream_output"),
            )

        seen_stream = False
        for mod in call.modifiers:
            if mod.name != "stream_output":
                suggestion = self._suggest(mod.name, ("stream_output",))
                patch = None
                if suggestion:
                    patch = replace_span_patch(
                        self.path, self.lines, mod.line, mod.column, len(mod.name), suggestion
                    )
                self.error(
                    "AE0302",
                    f"Unknown modifier '-> {mod.name}'",
                    mod.line,
                    mod.column,
                    hint="run accepts only '-> stream_output'.",
                    patch=patch,
                )
                continue
            if seen_stream:
                self.error(
                    "AE0302",
                    "Duplicate '-> stream_output' modifier",
                    mod.line,
                    mod.column,
                    hint="Remove the repeated modifier.",
                )
            seen_stream = True
            if mod.args:
                self.error(
                    "AE0212",
                    "stream_output takes no arguments",
                    mod.args[0].line,
                    mod.args[0].column,
                    hint="Write '-> stream_output' without parentheses.",
                )

        if model_ty is None or data_ty is None:
            return ERROR
        return Ty("stream")

    # -------------------------------------------------------------- primitives
    def parse_policy(self, mod: Modifier) -> dict | None:
        if mod.name != "optimize":
            return None
        policy = {"speed": "balanced"}
        seen = set()
        valid = True

        for arg in mod.args:
            if arg.name is None:
                self.error(
                    "AE0212",
                    "optimize arguments must be named",
                    arg.line,
                    arg.column,
                    hint="Write '-> optimize(speed: max)'.",
                )
                valid = False
                continue
            if arg.name != "speed":
                suggestion = self._suggest(arg.name, ("speed",))
                patch = None
                if suggestion:
                    patch = replace_span_patch(
                        self.path, self.lines, arg.line, arg.column, len(arg.name), suggestion
                    )
                self.error(
                    "AE0212",
                    f"Unknown optimize option '{arg.name}'",
                    arg.line,
                    arg.column,
                    hint="v0.1 supports only 'speed: max|balanced|low'.",
                    patch=patch,
                )
                valid = False
                continue
            if arg.name in seen:
                self.error(
                    "AE0212",
                    "Duplicate optimize option 'speed'",
                    arg.line,
                    arg.column,
                    hint="Keep a single speed value.",
                )
                valid = False
                continue
            seen.add(arg.name)

            value = arg.value
            text = None
            if isinstance(value, Literal) and isinstance(value.value, str):
                text = value.value
            elif isinstance(value, Name):
                text = value.id

            if text not in SPEED_VALUES:
                suggestion = self._suggest(text or "", SPEED_VALUES)
                hint = "speed must be one of: max, balanced, low."
                patch = None
                if suggestion and isinstance(value, Name):
                    hint = f"Did you mean 'speed: {suggestion}'?"
                    patch = replace_span_patch(
                        self.path,
                        self.lines,
                        value.line,
                        value.column,
                        len(value.id),
                        suggestion,
                    )
                self.error(
                    "AE0212",
                    f"Invalid optimize speed {text!r}",
                    value.line,
                    value.column,
                    hint=hint,
                    patch=patch,
                )
                valid = False
                continue
            policy["speed"] = text

        return policy if valid else None

    def bind(self, call: Call, spec: list[tuple[str, bool]]) -> tuple[dict[str, object], bool]:
        names = [name for name, _ in spec]
        bound: dict[str, object] = {}
        ok = True
        position = 0

        for arg in call.args:
            if arg.name is None:
                if position >= len(names):
                    self.error(
                        "AE0212",
                        f"Too many arguments for {call.name}()",
                        arg.line,
                        arg.column,
                        hint=f"Signature: {SIGNATURES[call.name]}",
                    )
                    ok = False
                    continue
                name = names[position]
                position += 1
            else:
                name = arg.name

            if name not in names:
                suggestion = self._suggest(name, names)
                hint = f"Signature: {SIGNATURES[call.name]}"
                patch = None
                if suggestion:
                    hint = f"Did you mean '{suggestion}'? Signature: {SIGNATURES[call.name]}"
                    patch = replace_span_patch(
                        self.path, self.lines, arg.line, arg.column, len(name), suggestion
                    )
                self.error(
                    "AE0212",
                    f"Unknown argument '{name}' for {call.name}()",
                    arg.line,
                    arg.column,
                    hint=hint,
                    patch=patch,
                )
                ok = False
                continue

            if name in bound:
                self.error(
                    "AE0212",
                    f"Duplicate argument '{name}'",
                    arg.line,
                    arg.column,
                    hint="Remove one of the duplicate arguments.",
                )
                ok = False
                continue

            bound[name] = arg.value

        for name, required in spec:
            if required and name not in bound:
                self.error(
                    "AE0212",
                    f"Missing required argument '{name}' for {call.name}()",
                    call.line,
                    call.column,
                    hint=f"Signature: {SIGNATURES[call.name]}",
                )
                ok = False

        return bound, ok

    def int_arg(self, node: object, label: str) -> int | None:
        if node is None:
            return None
        if isinstance(node, Literal) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value

        hint = f"{label} must be an integer literal, e.g. {label}: 1024."
        if isinstance(node, Name):
            hint = f"'{node.id}' cannot be used as {label} in v0.1; write an integer literal."
        self.error(
            "AE0212",
            f"{label} must be an integer literal",
            getattr(node, "line", 1),
            getattr(node, "column", 1),
            hint=hint,
        )
        return None

    def as_tensor(self, node: object, scope: dict[str, Ty], label: str) -> Ty | None:
        if isinstance(node, Name):
            ty = self.lookup(node, scope)
            if ty.kind == "error":
                return None
            if ty.kind != "tensor":
                self.error(
                    "AE0210",
                    f"'{label}' must be a tensor, but '{node.id}' is {self.article(ty.kind)}",
                    node.line,
                    node.column,
                    hint="Allocate it with 'tensor(rows, cols) -> auto_hardware'.",
                )
                return None
            return ty

        if isinstance(node, Literal) and isinstance(node.value, (int, float, str)):
            self.error(
                "AE0210",
                f"'{label}' must be a tensor, not a literal",
                node.line,
                node.column,
                hint="Pass a tensor variable allocated with tensor(rows, cols) -> auto_hardware.",
            )
            return None

        self.error(
            "AE0210",
            f"'{label}' must be a tensor",
            getattr(node, "line", 1),
            getattr(node, "column", 1),
            hint="Pass a tensor variable allocated with tensor(rows, cols) -> auto_hardware.",
        )
        return None

    @staticmethod
    def article(kind: str) -> str:
        if kind == "error":
            return "invalid"
        return f"a {kind}" if kind in {"dataset", "model", "tensor", "stream"} else f"an {kind}"

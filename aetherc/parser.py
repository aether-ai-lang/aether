"""Recursive-descent parser for Aether .ae programs."""

from __future__ import annotations

from .ast_nodes import Arg, Call, Literal, Modifier, Name, Program, Stmt, Task, Use
from .diagnostics import AetherError, replace_line_patch
from .lexer import Token


class Parser:
    def __init__(self, tokens: list[Token], path: str, source: str) -> None:
        self.tokens = tokens
        self.path = path
        self.source = source
        self.lines = source.splitlines()
        self.index = 0

    # ------------------------------------------------------------------ utils
    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.index + offset, len(self.tokens) - 1)]

    def advance(self) -> Token:
        token = self.peek()
        if token.kind != "EOF":
            self.index += 1
        return token

    def at(self, kind: str, value: object | None = None) -> bool:
        token = self.peek()
        return token.kind == kind and (value is None or token.value == value)

    def at_keyword(self, word: str) -> bool:
        return self.peek().kind == "IDENT" and self.peek().value == word

    def describe(self, token: Token) -> str:
        if token.kind == "NEWLINE":
            return "end of line"
        if token.kind == "EOF":
            return "end of file"
        if token.kind == "INDENT":
            return "indent"
        if token.kind == "DEDENT":
            return "dedent"
        if token.kind == "STRING":
            return f"string {token.value!r}"
        return f"'{token.value}'"

    def expect(self, kind: str, value: object | None = None, hint: str | None = None) -> Token:
        token = self.peek()
        if token.kind != kind or (value is not None and token.value != value):
            wanted = f"'{value}'" if value is not None else kind.lower()
            raise AetherError(
                f"Expected {wanted}, found {self.describe(token)}",
                token.line,
                token.column,
                code="AE0100",
                hint=hint,
            )
        return self.advance()

    def _add_colon_patch(self, lineno: int) -> str | None:
        if not 1 <= lineno <= len(self.lines):
            return None
        line = self.lines[lineno - 1]
        if "#" in line:
            return None
        return replace_line_patch(self.path, self.lines, lineno, line.rstrip() + ":")

    # --------------------------------------------------------------- program
    def parse_program(self) -> Program:
        uses: list[Use] = []
        tasks: list[Task] = []

        while not self.at("EOF"):
            if self.at("NEWLINE"):
                self.advance()
                continue
            if self.at("INDENT"):
                token = self.peek()
                raise AetherError(
                    "Unexpected indentation at top level",
                    token.line,
                    token.column,
                    code="AE0100",
                    hint="Top-level statements must be 'use ...' or 'task ...' with no leading spaces.",
                )
            if self.at_keyword("use"):
                uses.append(self.parse_use())
                continue
            if self.at_keyword("task"):
                tasks.append(self.parse_task())
                continue

            token = self.peek()
            raise AetherError(
                f"Expected 'use' or 'task', found {self.describe(token)}",
                token.line,
                token.column,
                code="AE0100",
                hint="A program is a sequence of 'use ai.core' and 'task <Name>:' declarations.",
            )

        return Program(uses=uses, tasks=tasks, path=self.path, source=self.source, lines=self.lines)

    def parse_use(self) -> Use:
        start = self.expect("IDENT", "use")
        if not self.at("IDENT"):
            token = self.peek()
            raise AetherError(
                "Expected a module name after 'use'",
                token.line,
                token.column,
                code="AE0200",
                hint="Use 'use ai.core'.",
            )

        parts = [self.expect("IDENT").value]
        while self.at("PUNCT", "."):
            self.advance()
            parts.append(self.expect("IDENT", hint="Module paths look like 'ai.core'.").value)

        self.expect("NEWLINE", hint="A 'use' declaration occupies exactly one line.")
        return Use(".".join(str(part) for part in parts), start.line, start.column)

    def parse_task(self) -> Task:
        start = self.expect("IDENT", "task")
        name_token = self.expect("IDENT", hint="Example: task HelloAether:")
        modifier = self.parse_modifier() if self.at("ARROW") else None

        if not self.at("PUNCT", ":"):
            token = self.peek()
            raise AetherError(
                f"Expected ':' after task name, found {self.describe(token)}",
                token.line,
                token.column,
                code="AE0102",
                hint=f"Write 'task {name_token.value}:' (a colon ends the header).",
                fix_patch=self._add_colon_patch(token.line),
            )
        self.advance()
        self.expect("NEWLINE")

        if not self.at("INDENT"):
            token = self.peek()
            raise AetherError(
                "Task body must be indented",
                token.line,
                token.column,
                code="AE0100",
                hint="Indent each statement under the task by 4 spaces.",
            )
        self.advance()

        body: list[Stmt] = []
        while not self.at("DEDENT") and not self.at("EOF"):
            if self.at("NEWLINE"):
                self.advance()
                continue
            body.append(self.parse_stmt())

        if self.at("DEDENT"):
            self.advance()

        end_line = body[-1].line if body else start.line
        return Task(
            name=str(name_token.value),
            modifier=modifier,
            body=body,
            line=start.line,
            column=start.column,
            end_line=end_line,
        )

    # ------------------------------------------------------------- statements
    def parse_stmt(self) -> Stmt:
        token = self.peek()

        if self.at_keyword("train"):
            return self.parse_train()
        if self.at_keyword("export"):
            return self.parse_export()
        if self.at_keyword("serve"):
            return self.parse_serve()

        if self.at_keyword("emit") or self.at_keyword("return"):
            kind = "emit" if token.value == "emit" else "return"
            self.advance()
            expr = self.parse_expr()
            self.expect("NEWLINE", hint=f"'{kind}' takes one expression on the same line.")
            return Stmt(kind=kind, expr=expr, line=token.line, column=token.column)

        if token.kind == "IDENT" and self.peek(1).kind == "PUNCT" and self.peek(1).value == "=":
            name_token = self.advance()
            self.advance()  # '='
            expr = self.parse_expr()
            self.expect("NEWLINE")
            return Stmt(
                kind="assign",
                expr=expr,
                line=token.line,
                column=token.column,
                name=str(name_token.value),
            )

        expr = self.parse_expr()
        self.expect("NEWLINE")
        return Stmt(kind="expr", expr=expr, line=token.line, column=token.column)

    def parse_train(self) -> Stmt:
        start = self.expect("IDENT", "train")
        model = self.expect(
            "IDENT", hint="Use 'train <model> on <dataset>(epochs: N) -> self_heal'."
        )
        if not self.at_keyword("on"):
            token = self.peek()
            raise AetherError(
                f"Expected 'on' after model name, found {self.describe(token)}",
                token.line,
                token.column,
                code="AE0100",
                hint="Training syntax: train model on dataset(epochs: 5) -> self_heal",
            )
        self.advance()
        dataset = self.expect(
            "IDENT", hint="Name the dataset variable created by load_data(...)."
        )
        self.expect("PUNCT", "(")

        if not self.at_keyword("epochs"):
            token = self.peek()
            raise AetherError(
                f"Expected 'epochs:' in train(...), found {self.describe(token)}",
                token.line,
                token.column,
                code="AE0212",
                hint="Example: train model on dataset(epochs: 5) -> self_heal",
            )
        self.advance()
        self.expect("PUNCT", ":")
        number = self.expect("NUMBER", hint="Epochs must be an integer literal, e.g. epochs: 5.")
        if not isinstance(number.value, int):
            raise AetherError(
                "Epochs must be an integer",
                number.line,
                number.column,
                code="AE0212",
                hint="Use a whole number such as epochs: 5.",
            )
        self.expect("PUNCT", ")")

        modifiers: list[Modifier] = []
        while self.at("ARROW"):
            modifiers.append(self.parse_modifier())
        self.expect("NEWLINE")

        return Stmt(
            kind="train",
            expr=None,
            line=start.line,
            column=start.column,
            target=str(model.value),
            data=str(dataset.value),
            epochs=int(number.value),
            modifiers=modifiers,
        )

    def parse_export(self) -> Stmt:
        start = self.expect("IDENT", "export")
        model = self.expect("IDENT", hint='Use: export model -> "model.ae.bin"')
        self.expect(
            "ARROW",
            hint='Export syntax: export model -> "model.ae.bin"',
        )
        path = self.expect(
            "STRING", hint='The export destination must be a quoted path, e.g. "model.ae.bin".'
        )
        self.expect("NEWLINE")
        return Stmt(
            kind="export",
            expr=None,
            line=start.line,
            column=start.column,
            target=str(model.value),
            path=str(path.value),
        )

    def parse_serve(self) -> Stmt:
        start = self.expect("IDENT", "serve")
        model = self.expect(
            "IDENT", hint='Use: serve model on port(8080) -> route("/predict")'
        )
        if not self.at_keyword("on"):
            token = self.peek()
            raise AetherError(
                f"Expected 'on' after model name, found {self.describe(token)}",
                token.line,
                token.column,
                code="AE0100",
                hint='Serving syntax: serve model on port(8080) -> route("/predict")',
            )
        self.advance()
        port_word = self.expect(
            "IDENT", "port", hint='Serving syntax: serve model on port(8080) -> route("/predict")'
        )
        self.expect("PUNCT", "(")
        port = self.expect(
            "NUMBER",
            hint="The port must be an integer literal, e.g. port(8080).",
        )
        if not isinstance(port.value, int):
            raise AetherError(
                "Port must be an integer",
                port.line,
                port.column,
                code="AE0212",
                hint="Use a whole number such as port(8080).",
            )
        self.expect("PUNCT", ")")

        modifiers: list[Modifier] = []
        while self.at("ARROW"):
            modifiers.append(self.parse_modifier())
        self.expect("NEWLINE")

        return Stmt(
            kind="serve",
            expr=None,
            line=start.line,
            column=start.column,
            target=str(model.value),
            port=int(port.value),
            modifiers=modifiers,
        )

    # ------------------------------------------------------------ expressions
    def parse_expr(self) -> object:
        token = self.peek()
        node: object

        if token.kind == "NUMBER":
            self.advance()
            node = Literal(token.value, token.line, token.column, raw=str(token.value))
        elif token.kind == "STRING":
            self.advance()
            node = Literal(token.value, token.line, token.column)
        elif token.kind == "IDENT" and token.value == "run" and self.peek(1).kind == "IDENT":
            self.advance()
            target = self.expect("IDENT")
            args = self.parse_args()
            node = Call("run", str(target.value), args, [], token.line, token.column)
        elif token.kind == "IDENT":
            self.advance()
            if self.at("PUNCT", "("):
                node = Call(str(token.value), None, self.parse_args(), [], token.line, token.column)
            else:
                node = Name(str(token.value), token.line, token.column)
        else:
            raise AetherError(
                f"Expected an expression, found {self.describe(token)}",
                token.line,
                token.column,
                code="AE0100",
                hint="Use an intent call such as tensor(...), neural_net(...), or run model(data).",
            )

        while self.at("ARROW"):
            if not isinstance(node, Call):
                raise AetherError(
                    "Modifiers can only follow an intent call",
                    self.peek().line,
                    self.peek().column,
                    code="AE0100",
                    hint="Example: tensor(1024, 1024) -> auto_hardware",
                )
            node.modifiers.append(self.parse_modifier())

        return node

    def parse_args(self) -> list[Arg]:
        self.expect("PUNCT", "(", hint="Intent calls need parentheses, e.g. tensor(512, 512).")
        args: list[Arg] = []

        if not self.at("PUNCT", ")"):
            while True:
                args.append(self.parse_arg())
                if self.at("PUNCT", ","):
                    self.advance()
                    continue
                break

        self.expect("PUNCT", ")", hint="Close the argument list with ')'.")
        return args

    def parse_arg(self) -> Arg:
        token = self.peek()
        if (
            token.kind == "IDENT"
            and self.peek(1).kind == "PUNCT"
            and self.peek(1).value == ":"
        ):
            name_token = self.advance()
            self.advance()  # ':'
            value = self.parse_value()
            return Arg(str(name_token.value), value, name_token.line, name_token.column)

        value = self.parse_value()
        return Arg(None, value, token.line, token.column)

    def parse_value(self) -> object:
        token = self.peek()
        if token.kind == "NUMBER":
            self.advance()
            return Literal(token.value, token.line, token.column, raw=str(token.value))
        if token.kind == "STRING":
            self.advance()
            return Literal(token.value, token.line, token.column)
        if token.kind == "IDENT":
            self.advance()
            return Name(str(token.value), token.line, token.column)
        raise AetherError(
            f"Expected a value, found {self.describe(token)}",
            token.line,
            token.column,
            code="AE0212",
            hint="Intent arguments are integer/string literals or variable names in v0.1.",
        )

    def parse_modifier(self) -> Modifier:
        self.expect("ARROW")
        name_token = self.expect("IDENT", hint="Example: -> auto_hardware")
        args = self.parse_args() if self.at("PUNCT", "(") else []
        return Modifier(str(name_token.value), args, name_token.line, name_token.column)


def parse(tokens: list[Token], path: str, source: str) -> Program:
    return Parser(tokens, path, source).parse_program()

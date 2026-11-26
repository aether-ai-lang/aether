"""Aether lexer: indentation-aware tokens for the .ae surface syntax."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from .diagnostics import AetherError, replace_line_patch


@dataclass
class Token:
    kind: str  # IDENT | NUMBER | STRING | ARROW | PUNCT | NEWLINE | INDENT | DEDENT | EOF
    value: object
    line: int
    column: int

    def __str__(self) -> str:
        return f"{self.kind}({self.value!r})"


_TOKEN_RE = re.compile(
    r"""
      (?P<NUMBER>\d+(?:\.\d+)?)
    | (?P<STRING>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
    | (?P<ARROW>->)
    | (?P<OP>==|!=|>=|<=)
    | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<PUNCT>[=:,()\[\].+\-*/%<>])
    """,
    re.VERBOSE,
)


def tokenize(source: str, path: str = "<aether>") -> list[Token]:
    lines = source.splitlines()
    tokens: list[Token] = []
    indent_stack = [0]

    for lineno, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        left = raw.lstrip(" \t")
        if left.startswith("#"):
            continue

        leading = raw[: len(raw) - len(left)]
        if "\t" in leading:
            column = leading.index("\t") + 1
            raise AetherError(
                "Tab character used for indentation",
                lineno,
                column,
                code="AE0002",
                hint="Aether blocks use spaces. Replace tab indentation with 4 spaces.",
                fix_patch=replace_line_patch(
                    path, lines, lineno, leading.expandtabs(4) + raw[len(leading) :]
                ),
            )

        indent = len(leading)
        if indent > indent_stack[-1]:
            indent_stack.append(indent)
            tokens.append(Token("INDENT", "", lineno, 1))
        elif indent < indent_stack[-1]:
            while indent < indent_stack[-1]:
                indent_stack.pop()
                tokens.append(Token("DEDENT", "", lineno, 1))
            if indent != indent_stack[-1]:
                raise AetherError(
                    "Unindent does not match any outer indentation level",
                    lineno,
                    1,
                    code="AE0003",
                    hint="Align this line with an enclosing block (multiples of 4 spaces).",
                )

        pos = 0
        while pos < len(raw):
            char = raw[pos]
            if char in " \t":
                pos += 1
                continue
            if char == "#":
                break

            match = _TOKEN_RE.match(raw, pos)
            if not match:
                raise AetherError(
                    f"Unexpected character {char!r}",
                    lineno,
                    pos + 1,
                    code="AE0001",
                    hint=(
                        "Aether supports literals, identifiers, '->', arithmetic "
                        "(+ - * / %), comparisons (== != > < >= <=), and ( ) , : = ."
                    ),
                )

            kind = match.lastgroup or "PUNCT"
            text = match.group()
            column = pos + 1
            if kind == "OP":
                tokens.append(Token("PUNCT", text, lineno, column))
            elif kind == "STRING":
                try:
                    value = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    raise AetherError(
                        "Invalid string literal",
                        lineno,
                        column,
                        code="AE0100",
                        hint="Close the quote and escape inner quotes with a backslash.",
                    ) from None
                tokens.append(Token("STRING", value, lineno, column))
            elif kind == "NUMBER":
                value: object = float(text) if "." in text else int(text)
                tokens.append(Token("NUMBER", value, lineno, column))
            else:
                tokens.append(Token(kind, text, lineno, column))
            pos = match.end()

        tokens.append(Token("NEWLINE", "", lineno, len(raw) + 1))

    eof_line = max(1, len(lines))
    for _ in indent_stack[1:]:
        tokens.append(Token("DEDENT", "", eof_line, 1))
    tokens.append(Token("EOF", "", eof_line, 1))
    return tokens

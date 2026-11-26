"""AST node definitions shared by the parser, checker, and code generator."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Use:
    module: str
    line: int
    column: int


@dataclass
class Literal:
    value: object
    line: int
    column: int
    raw: str = ""


@dataclass
class Name:
    id: str
    line: int
    column: int


@dataclass
class Arg:
    name: str | None
    value: object
    line: int
    column: int


@dataclass
class Modifier:
    name: str
    args: list[Arg]
    line: int
    column: int


@dataclass
class Call:
    name: str
    target: str | None
    args: list[Arg]
    modifiers: list[Modifier]
    line: int
    column: int


@dataclass
class Stmt:
    kind: str  # assign | emit | return | expr | train | export
    expr: object | None
    line: int
    column: int
    name: str | None = None
    target: str | None = None
    data: str | None = None
    epochs: int | None = None
    path: str | None = None
    modifiers: list[Modifier] = field(default_factory=list)


@dataclass
class Task:
    name: str
    modifier: Modifier | None
    body: list[Stmt]
    line: int
    column: int
    end_line: int = 0


@dataclass
class Program:
    uses: list[Use]
    tasks: list[Task]
    path: str
    source: str
    lines: list[str] = field(default_factory=list)

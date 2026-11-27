"""Diagnostics, self-healing patch generation, and error plumbing."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

AETHER_VERSION = "1.0.0"


@dataclass
class Diagnostic:
    """A single machine-readable compiler/runtime diagnostic."""

    code: str
    message: str
    line: int = 1
    column: int = 1
    severity: str = "error"
    hint: str | None = None
    fix_patch: str | None = None

    def to_dict(self) -> dict:
        data: dict = {
            "code": self.code,
            "severity": self.severity,
            "line": self.line,
            "column": self.column,
            "message": self.message,
        }
        if self.hint:
            data["hint"] = self.hint
        if self.fix_patch:
            data["fix_patch"] = self.fix_patch
        return data


def unified_patch(path: str, before: list[str], after: list[str]) -> str | None:
    """Build a git-applicable unified diff, or None when nothing changed."""
    normalized = path.replace("\\", "/")
    diff = difflib.unified_diff(
        before,
        after,
        fromfile=f"a/{normalized}",
        tofile=f"b/{normalized}",
        n=2,
        lineterm="",
    )
    lines = list(diff)
    if not lines:
        return None
    return "\n".join(lines) + "\n"


def replace_line_patch(path: str, lines: list[str], lineno: int, new_text: str) -> str | None:
    if not 1 <= lineno <= len(lines):
        return None
    after = list(lines)
    after[lineno - 1] = new_text
    return unified_patch(path, lines, after)


def replace_span_patch(
    path: str, lines: list[str], lineno: int, column: int, length: int, new_text: str
) -> str | None:
    if not 1 <= lineno <= len(lines):
        return None
    source = lines[lineno - 1]
    start = column - 1
    if start < 0 or start + length > len(source):
        return None
    after = list(lines)
    after[lineno - 1] = source[:start] + new_text + source[start + length :]
    return unified_patch(path, lines, after)


def insert_line_patch(path: str, lines: list[str], before_lineno: int, new_text: str) -> str | None:
    index = max(0, min(len(lines), before_lineno - 1))
    after = list(lines)
    after.insert(index, new_text)
    return unified_patch(path, lines, after)


def delete_line_patch(path: str, lines: list[str], lineno: int) -> str | None:
    if not 1 <= lineno <= len(lines):
        return None
    after = list(lines)
    del after[lineno - 1]
    return unified_patch(path, lines, after)


def _patch_edits(patch_text: str, original: list[str]) -> list[tuple[int, int, list[str]]] | None:
    """Translate a unified diff into minimal edits against original coordinates."""
    hunks: list[tuple[int, list[str], list[str]]] = []
    old_start = 0
    old_block: list[str] = []
    new_block: list[str] = []
    in_hunk = False

    for raw in patch_text.splitlines():
        if raw.startswith("@@"):
            if in_hunk:
                hunks.append((old_start, old_block, new_block))
            match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", raw)
            if not match:
                return None
            old_start = int(match.group(1))
            old_block, new_block, in_hunk = [], [], True
            continue
        if raw.startswith("--- ") or raw.startswith("+++ "):
            continue
        if not in_hunk:
            continue
        if raw.startswith("\\"):
            continue
        if raw.startswith("+"):
            new_block.append(raw[1:])
        elif raw.startswith("-"):
            old_block.append(raw[1:])
        elif raw.startswith(" "):
            old_block.append(raw[1:])
            new_block.append(raw[1:])
        elif raw == "":
            old_block.append("")
            new_block.append("")

    if in_hunk:
        hunks.append((old_start, old_block, new_block))

    edits: list[tuple[int, int, list[str]]] = []
    for start, old, new in hunks:
        index = max(0, start - 1)
        if original[index : index + len(old)] != old:
            return None
        matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            edits.append((index + i1, index + i2, new[j1:j2]))
    return edits


def merged_fix_patch(path: str, lines: list[str], diagnostics: list) -> str | None:
    """Compose every safe diagnostic repair into one reviewable patch."""
    edits: list[tuple[int, int, list[str]]] = []
    for diagnostic in diagnostics:
        if not diagnostic.fix_patch:
            continue
        parsed = _patch_edits(diagnostic.fix_patch, lines)
        if parsed is None:
            continue
        for start, end, replacement in parsed:
            overlaps = any(
                not (end <= other_start or start >= other_end)
                for other_start, other_end, _ in edits
            )
            if overlaps:
                continue
            edits.append((start, end, replacement))

    if not edits:
        return None

    result: list[str] = []
    cursor = 0
    for start, end, replacement in sorted(edits, key=lambda item: (item[0], item[1])):
        if start < cursor:
            continue
        result.extend(lines[cursor:start])
        result.extend(replacement)
        cursor = end
    result.extend(lines[cursor:])
    return unified_patch(path, lines, result)


def apply_patch(patch_text: str, source: str) -> str | None:
    """Apply one of Aether's unified diffs to source text.

    Returns the patched source, or None when the patch does not apply
    (context mismatch). This powers `aether mcp` self-healing.
    """
    lines = source.splitlines()
    edits = _patch_edits(patch_text, lines)
    if not edits:
        return None
    result: list[str] = []
    cursor = 0
    for start, end, replacement in sorted(edits, key=lambda item: (item[0], item[1])):
        if start < cursor:
            return None
        result.extend(lines[cursor:start])
        result.extend(replacement)
        cursor = end
    result.extend(lines[cursor:])
    trailing = "\n" if source.endswith("\n") else ""
    return "\n".join(result) + trailing


class AetherError(Exception):
    """Lexer/parser failure carrying a ready-to-serialize diagnostic."""

    def __init__(
        self,
        message: str,
        line: int = 1,
        column: int = 1,
        code: str = "AE0100",
        hint: str | None = None,
        fix_patch: str | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic = Diagnostic(
            code=code,
            message=message,
            line=line,
            column=column,
            hint=hint,
            fix_patch=fix_patch,
        )

"""Helpers for building Numba-compatible fixed-size tuple callbacks."""

from __future__ import annotations


def make_tuple_callback(name: str, body_lines: list[str], values: list[str]):
    source = [f"def {name}(y, t, p):"]
    source.extend(f"    {line}" for line in body_lines)
    source.append("    return (")
    source.extend(f"        ({value}) + 0.0," for value in values)
    source.append("    )")
    namespace: dict[str, object] = {}
    exec("\n".join(source), namespace)
    return namespace[name]


def make_matrix_callback(name: str, body_lines: list[str], rows: list[list[str]]):
    source = [f"def {name}(y, t, p):"]
    source.extend(f"    {line}" for line in body_lines)
    source.append("    return (")
    for row in rows:
        source.append("        (")
        source.extend(f"            ({value}) + 0.0," for value in row)
        source.append("        ),")
    source.append("    )")
    namespace: dict[str, object] = {}
    exec("\n".join(source), namespace)
    return namespace[name]

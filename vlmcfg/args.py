"""Command-line `--model_args` into frozen config dataclasses."""

from dataclasses import fields
from typing import Any, TypeVar

T = TypeVar("T")


def split_args(cls: type[T], args: dict[str, Any]) -> tuple[T, dict[str, Any]]:
    """Build `cls` from the entries of `args` named after its fields; return the rest.

    Values arrive as strings or as lmms-eval's parsed scalars; each is converted to its
    field's type ("true"/"1"/"yes" for booleans).
    """
    types = {f.name: f.type for f in fields(cls)}  # type: ignore[arg-type]
    mine = {k: _convert(types[k], v) for k, v in args.items() if k in types}
    rest = {k: v for k, v in args.items() if k not in types}
    return cls(**mine), rest


def _convert(kind: Any, value: Any) -> Any:
    if kind is bool and isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return kind(value)

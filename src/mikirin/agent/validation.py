from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_TYPE_CHECKS = {
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "array": lambda value: isinstance(value, list),
    "object": lambda value: isinstance(value, Mapping),
}


def validate_tool_arguments(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> None:
    if schema.get("type") not in {None, "object"}:
        raise ValueError("tool parameter schema must describe an object")
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ValueError("tool parameter properties must be an object")
    required = schema.get("required", [])
    for name in required:
        if name not in arguments:
            raise ValueError(f"missing required argument: {name}")
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ValueError(f"unknown arguments: {', '.join(unknown)}")
    for name, value in arguments.items():
        rule = properties.get(name)
        if not isinstance(rule, Mapping):
            continue
        expected = rule.get("type")
        check = _TYPE_CHECKS.get(expected)
        if check is not None and not check(value):
            raise ValueError(f"argument {name!r} must be {expected}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in rule and value < rule["minimum"]:
                raise ValueError(f"argument {name!r} is below minimum")
            if "maximum" in rule and value > rule["maximum"]:
                raise ValueError(f"argument {name!r} is above maximum")


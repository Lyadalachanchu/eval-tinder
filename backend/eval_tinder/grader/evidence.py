"""Evidence pointer validation.

An evidence item is ``{"pointer": "/tool_calls/0/result/status", "quote": "accepted"}``.
The pointer must resolve inside the case document and the quote must appear in
the resolved value's string form. Invalid evidence never becomes PASS.
"""
from __future__ import annotations

import json
from typing import Any


class EvidenceError(ValueError):
    pass


def resolve_pointer(data: Any, pointer: str) -> Any:
    if pointer == "":
        return data
    if not pointer.startswith("/"):
        raise EvidenceError(f"Pointer must start with '/': {pointer!r}")
    node = data
    for raw in pointer.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            if token not in node:
                raise EvidenceError(f"Pointer {pointer!r}: key {token!r} not found")
            node = node[token]
        elif isinstance(node, list):
            try:
                idx = int(token)
            except ValueError as e:
                raise EvidenceError(f"Pointer {pointer!r}: {token!r} is not an index") from e
            if idx < 0 or idx >= len(node):
                raise EvidenceError(f"Pointer {pointer!r}: index {idx} out of range")
            node = node[idx]
        else:
            raise EvidenceError(f"Pointer {pointer!r}: cannot descend into scalar at {token!r}")
    return node


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_evidence_json(raw: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as e:
        raise EvidenceError(f"evidence_json is not valid JSON: {e.msg}") from e
    if not isinstance(parsed, list):
        raise EvidenceError("evidence_json must be a JSON list")
    items: list[dict[str, Any]] = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict) or "pointer" not in item:
            raise EvidenceError(f"evidence item {i} must be an object with a 'pointer'")
        pointer = item["pointer"]
        quote = item.get("quote", "")
        if not isinstance(pointer, str) or not isinstance(quote, str):
            raise EvidenceError(f"evidence item {i}: pointer and quote must be strings")
        items.append({"pointer": pointer, "quote": quote})
    return items


def validate_evidence(items: list[dict[str, Any]], case_data: Any, *, max_items: int = 12) -> list[dict[str, Any]]:
    if len(items) > max_items:
        raise EvidenceError(f"too many evidence items ({len(items)} > {max_items})")
    validated = []
    for item in items:
        value = resolve_pointer(case_data, item["pointer"])
        text = _as_text(value)
        quote = item["quote"]
        if quote and quote.strip() not in text:
            raise EvidenceError(f"quote {quote[:60]!r} not found at {item['pointer']!r}")
        validated.append({"pointer": item["pointer"], "quote": quote})
    return validated

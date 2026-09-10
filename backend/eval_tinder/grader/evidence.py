"""Evidence pointer validation.

An evidence item is ``{"pointer": "/tool_calls/0/result/status", "quote": "accepted"}``.
The pointer must resolve inside the case document and the quote must appear in
the resolved value's string form. Invalid evidence never becomes PASS.
"""
from __future__ import annotations

import json
import re
from typing import Any


class EvidenceError(ValueError):
    pass


# Section labels a model may echo instead of the data keys (older renderer names, upper-case variants).
_KEY_ALIASES = {
    "tool_calls_json": "tool_calls",
    "toolcalls": "tool_calls",
    "target_output": "output",
    "user_request": "input",
    "context_json": "context",
    "task_metadata_json": "metadata",
    "task_metadata": "metadata",
}
_TOP_LEVEL_KEYS = ("input", "context", "tool_calls", "output", "metadata")


def normalize_pointer(pointer: str) -> str:
    """Canonicalize common pointer spellings into an RFC 6901 pointer.

    Accepts dotted/bracket paths (``tool_calls[0].result.status``), missing leading
    slashes, and known section aliases (``/TOOL_CALLS_JSON/0``). This only rewrites
    syntax; it never changes which value the pointer resolves to.
    """
    if not isinstance(pointer, str):
        raise EvidenceError("pointer must be a string")
    text = pointer.strip()
    if text in ("", "/"):
        return text
    if text.startswith("#/"):
        text = text[1:]
    if not text.startswith("/"):
        text = re.sub(r"\[(\d+)\]", r"/\1", text)
        text = "/" + text.replace(".", "/")
    text = text.replace("//", "/")
    tokens = text.split("/")[1:]
    if tokens:
        first = tokens[0]
        lowered = first.lower()
        if lowered in _KEY_ALIASES:
            tokens[0] = _KEY_ALIASES[lowered]
        elif lowered in _TOP_LEVEL_KEYS:
            tokens[0] = lowered
    return "/" + "/".join(tokens)


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


def parse_evidence_json(raw: str) -> list[dict[str, Any]]:  # noqa: C901
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
        items.append({"pointer": normalize_pointer(pointer), "quote": quote})
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

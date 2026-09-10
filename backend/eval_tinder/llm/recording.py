"""A transparent recording wrapper for any legacy-contract LM (used by leakage tests)."""
from __future__ import annotations

import threading
from typing import Any

from dspy.clients.base_lm import BaseLM


class RecordingLM(BaseLM):
    forward_contract = "legacy"

    def __init__(self, inner: BaseLM):
        super().__init__(
            model=inner.model, model_type=getattr(inner, "model_type", "chat"),
            temperature=inner.kwargs.get("temperature", 0.0), max_tokens=inner.kwargs.get("max_tokens", 4000), cache=False,
        )
        self.kwargs = dict(inner.kwargs)
        self.inner = inner
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def forward(self, prompt=None, messages=None, **kwargs):
        response = self.inner.forward(prompt=prompt, messages=messages, **kwargs)
        msgs = messages or [{"role": "user", "content": prompt or ""}]
        text = ""
        try:
            text = response.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            self.calls.append({"messages": msgs, "output": text})
        return response

    async def aforward(self, prompt=None, messages=None, **kwargs):
        return self.forward(prompt=prompt, messages=messages, **kwargs)

    def all_prompt_text(self) -> str:
        with self._lock:
            return "\n".join(str(m.get("content", "")) for c in self.calls for m in c["messages"])

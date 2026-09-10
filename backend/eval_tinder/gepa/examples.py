"""Build ``dspy.Example`` objects from frozen snapshot items.

Model inputs are exactly ``project_context`` and ``case``; everything else on the
example (expert label, explanation, partition, case_data, index) is bookkeeping
that only the metric may read. ``with_inputs`` enforces the split. [S5]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import dspy

from eval_tinder.domain.rendering import CaseDocument


@dataclass(frozen=True)
class LabeledCase:
    """A frozen (trace, human judgment) pair. ``label`` must be PASS or FAIL for optimization."""

    index: int
    case: CaseDocument
    label: str
    explanation: str
    partition: str


def build_examples(project_context: str, cases: list[LabeledCase]) -> list[dspy.Example]:
    examples = []
    for c in cases:
        if c.label not in {"PASS", "FAIL"}:
            raise ValueError(f"Unresolved judgment {c.label!r} cannot enter binary optimization (case {c.index})")
        ex = dspy.Example(
            project_context=project_context,
            case=c.case.text,
            expert_grade=c.label,
            expert_explanation=c.explanation or "",
            partition=c.partition,
            case_data=c.case.data,
            case_index=c.index,
            case_hash=c.case.text_hash,
        ).with_inputs("project_context", "case")
        examples.append(ex)
    return examples


def example_inputs_only(example: dspy.Example) -> dict[str, Any]:
    """What the model may see for this example."""
    return dict(example.inputs())

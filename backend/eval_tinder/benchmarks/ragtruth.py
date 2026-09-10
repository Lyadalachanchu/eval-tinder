"""RAGTruth -> eval-tinder import files.

RAGTruth (Niu et al., ACL 2024) holds ~18k RAG answers from six models with human
span-level hallucination annotations. This module turns a seeded sample of the
train split into the app's JSONL import format plus a truth table, and the
official test split into benchmark files that never enter the application.

Label policy (binary, response level): any annotated span = FAIL, none = PASS,
a response flagged "truncated" = CANNOT_JUDGE (OTHER). The annotators' notes
become the expert explanations.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

HF_PARQUET = {
    "train": "https://huggingface.co/datasets/wandb/RAGTruth-processed/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "test": "https://huggingface.co/datasets/wandb/RAGTruth-processed/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
}
TASK_TYPES = ("QA", "Summary", "Data2txt")
PROJECT_DESCRIPTION = (
    "A retrieval-augmented assistant. For each case it received an instruction or question together with "
    "source material (retrieved passages, a news article, or a structured business record) and wrote an answer. "
    "The expert judges whether the answer is fully supported by the supplied source material."
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


def _parse_labels(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    text = str(raw)
    for candidate in (text, text.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")):
        try:
            parsed = json.loads(candidate)
            return [x for x in parsed if isinstance(x, dict)] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            continue
    return []


def source_group(row: dict[str, Any]) -> str:
    digest = hashlib.sha256((str(row["query"]) + "\n" + str(row["context"])).encode("utf-8")).hexdigest()[:16]
    return f"rt-src-{digest}"


def to_record(row: dict[str, Any], split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = _parse_labels(row.get("hallucination_labels"))
    quality = str(row.get("quality") or "good")
    if quality == "truncated":
        verdict, reason = "CANNOT_JUDGE", "OTHER"
        explanation = "The recorded answer is truncated; the annotators could not judge it as a whole."
    elif labels:
        verdict, reason = "FAIL", None
        notes = []
        for lab in labels[:4]:
            meta = str(lab.get("meta") or lab.get("label_type") or "").strip().replace("\n", " ")
            notes.append(f"[{lab.get('label_type')}] {meta}"[:220])
        explanation = "Unsupported or contradicting content relative to the source. " + " | ".join(notes)
    else:
        verdict, reason = "PASS", None
        explanation = "Every claim in the answer is supported by the supplied source material."
        if quality == "incorrect_refusal":
            explanation += " (The answer refused although the source contained the answer; grounding is still intact.)"
    external_id = f"rt-{split}-{row['id']}"
    record = {
        "external_id": external_id,
        "group_id": source_group(row),
        "timestamp": "2023-12-01T00:00:00Z",
        "input": str(row["query"]),
        "context": {"source_material": str(row["context"])},
        "tool_calls": [],
        "output": str(row["output"]),
        "metadata": {
            "task_type": str(row["task_type"]),
            "language": "en",
            "channel": "rag",
            "author_model": str(row.get("model")),
            "quality_flag": quality,
            "benchmark": "RAGTruth",
        },
        "source_type": "PRODUCTION",
    }
    truth = {
        "verdict": verdict,
        "cannot_judge_reason": reason,
        "explanation": explanation,
        "task_type": str(row["task_type"]),
        "author_model": str(row.get("model")),
        "n_spans": len(labels),
        "span_types": sorted({str(lab.get("label_type")) for lab in labels}),
        "quality_flag": quality,
    }
    return record, truth


def sample_train(rows: list[dict[str, Any]], *, groups_per_task: int, seed: int) -> list[dict[str, Any]]:
    """Seeded, task-stratified sample of whole source groups (all six answers per source)."""
    by_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_task[str(row["task_type"])][source_group(row)].append(row)
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    for task in TASK_TYPES:
        groups = sorted(by_task[task])
        rng.shuffle(groups)
        for gid in groups[:groups_per_task]:
            picked.extend(by_task[task][gid])
    return picked


def subsample_test(records: list[dict[str, Any]], *, groups_per_task: int, seed: int) -> list[dict[str, Any]]:
    by_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        by_task[rec["metadata"]["task_type"]][rec["group_id"]].append(rec)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for task in TASK_TYPES:
        groups = sorted(by_task[task])
        rng.shuffle(groups)
        for gid in groups[:groups_per_task]:
            out.extend(by_task[task][gid])
    return out


def prepare(
    out_dir: Path,
    *,
    train_parquet: Path,
    test_parquet: Path,
    groups_per_task: int = 100,
    test_subsample_groups_per_task: int = 34,
    seed: int = 2026,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _load_rows(train_parquet)
    test_rows = _load_rows(test_parquet)
    sampled = sample_train(train_rows, groups_per_task=groups_per_task, seed=seed)
    train_records, truth = [], {}
    for row in sampled:
        rec, t = to_record(row, "train")
        train_records.append(rec)
        truth[rec["external_id"]] = t
    test_records, test_truth = [], {}
    for row in test_rows:
        rec, t = to_record(row, "test")
        test_records.append(rec)
        test_truth[rec["external_id"]] = t
    sub = subsample_test(test_records, groups_per_task=test_subsample_groups_per_task, seed=seed + 1)
    (out_dir / "train_sample.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_records))
    (out_dir / "train_truth.json").write_text(json.dumps({"policy": "RAGTruth grounding: FAIL when any span is unsupported by or contradicts the source; PASS otherwise; truncated answers are CANNOT_JUDGE.", "labels": truth}, indent=1))
    (out_dir / "test_full.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in test_records))
    (out_dir / "test_subsample.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sub))
    (out_dir / "test_truth.json").write_text(json.dumps({"labels": test_truth}, indent=1))

    def _stats(recs: list[dict[str, Any]], tr: dict[str, Any]) -> dict[str, Any]:
        counts: dict[str, Any] = {"records": len(recs), "groups": len({r["group_id"] for r in recs}), "by_verdict": {}, "by_task": {}}
        for r in recs:
            v = tr[r["external_id"]]["verdict"]
            counts["by_verdict"][v] = counts["by_verdict"].get(v, 0) + 1
            task = r["metadata"]["task_type"]
            d = counts["by_task"].setdefault(task, {"records": 0, "FAIL": 0})
            d["records"] += 1
            d["FAIL"] += int(v == "FAIL")
        return counts

    manifest = {
        "seed": seed,
        "groups_per_task": groups_per_task,
        "train_sample": _stats(train_records, truth),
        "test_full": _stats(test_records, test_truth),
        "test_subsample": _stats(sub, test_truth),
        "source": HF_PARQUET,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest

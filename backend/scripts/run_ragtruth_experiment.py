"""Launch the RAGTruth experiment (random versus committee selection) against the configured provider.

Usage (from backend/): GRADING_CONCURRENCY=8 MAX_PROVIDER_CALLS_PER_JOB=3000 uv run python scripts/run_ragtruth_experiment.py
The driver is resumable: rerunning with the same --out picks up from state.json.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
logging.basicConfig(level=logging.WARNING)
logging.getLogger("dspy").setLevel(logging.ERROR)
logging.getLogger("gepa").setLevel(logging.ERROR)

from eval_tinder.benchmarks.ragtruth_experiment import run  # noqa: E402
from eval_tinder.config import get_settings  # noqa: E402
from eval_tinder.db.base import get_session_factory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/ragtruth"), help="output of eval_tinder.benchmarks.ragtruth.prepare")
    parser.add_argument("--out", type=Path, default=Path("artifacts/ragtruth_experiment"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--metric-calls", type=int, default=300)
    parser.add_argument("--score-concurrency", type=int, default=8)
    parser.add_argument("--strategies", default="random,committee")
    args = parser.parse_args()
    settings = get_settings()
    if settings.llm_provider != "litellm":
        raise SystemExit("set LLM_PROVIDER=litellm plus GRADER_MODEL/REFLECTION_MODEL; the fake provider cannot grade RAGTruth")
    print(f"provider {settings.llm_provider} {settings.grader_model} concurrency {settings.grading_concurrency} "
          f"max calls/job {settings.max_provider_calls_per_job}", flush=True)
    started = time.time()
    run(get_session_factory(), data_dir=args.data, out_dir=args.out, seed=args.seed, metric_calls=args.metric_calls,
        settings=settings, score_concurrency=args.score_concurrency, strategies=tuple(args.strategies.split(",")))
    print("FINISHED in %.0f min" % ((time.time() - started) / 60), flush=True)


if __name__ == "__main__":
    main()

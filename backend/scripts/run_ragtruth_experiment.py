import logging, os, time
from pathlib import Path
os.environ.setdefault("TQDM_DISABLE", "1")
logging.basicConfig(level=logging.WARNING)
logging.getLogger("dspy").setLevel(logging.ERROR); logging.getLogger("gepa").setLevel(logging.ERROR)
from eval_tinder.config import get_settings
from eval_tinder.db.base import get_session_factory
from eval_tinder.benchmarks.ragtruth_experiment import run
s = get_settings()
assert s.llm_provider == "litellm", s.llm_provider
print("provider", s.llm_provider, s.grader_model, "concurrency", s.grading_concurrency, "max calls/job", s.max_provider_calls_per_job, flush=True)
t0 = time.time()
res = run(get_session_factory(), data_dir=Path("artifacts/ragtruth"), out_dir=Path("artifacts/ragtruth_experiment"), seed=1, settings=s, score_concurrency=8)
print("FINISHED in %.0f min" % ((time.time() - t0) / 60), flush=True)

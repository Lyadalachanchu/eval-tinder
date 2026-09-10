FROM python:3.11-slim
RUN pip install --no-cache-dir uv==0.8.17
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uv", "run", "uvicorn", "eval_tinder.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

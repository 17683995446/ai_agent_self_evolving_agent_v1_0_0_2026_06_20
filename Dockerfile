FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/

RUN mkdir -p data logs snapshots

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.cli.main", "--config", "config/config.prod.yaml", "start"]

FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY ayabada ./ayabada
COPY data ./data
RUN pip install --no-cache-dir ".[llm]"

# Heartbeat state and handoff docs live on a mounted volume.
VOLUME ["/data"]

ENTRYPOINT ["ayabada"]
CMD ["demo", "--out", "/data/handoff-demo.md"]

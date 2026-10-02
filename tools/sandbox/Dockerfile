FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir httpx pytest
# Runtime source is mounted read-only, not copied from the user's workspace.
WORKDIR /workspace

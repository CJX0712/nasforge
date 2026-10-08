# NasForge Dockerfile
# Build is fully offline: no model downloads, no network calls at build or run time.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY nasforge/ nasforge/

RUN pip install -e ".[dev]"

ENTRYPOINT ["python", "-m", "nasforge.cli"]
CMD ["demo"]

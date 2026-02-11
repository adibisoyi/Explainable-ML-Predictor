FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir .

COPY artifacts /app/artifacts

EXPOSE 8000
CMD ["python", "-m", "exml.cli", "serve", "--host", "0.0.0.0", "--port", "8000", "--artifacts", "artifacts"]

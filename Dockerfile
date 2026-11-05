FROM python:3.12-slim

WORKDIR /app

# Dependency layer first so code-only changes rebuild quickly.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

ARG APP_VERSION=dev
ENV APP_VERSION=$APP_VERSION PORT=8000
EXPOSE 8000

# Unprivileged: the server needs no filesystem or network access beyond its port.
RUN useradd --create-home --uid 10001 app
USER app

CMD ["glen-mcp-opt"]

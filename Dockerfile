FROM python:3.12-slim

WORKDIR /app
RUN useradd --create-home --uid 10001 app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# DuckDB extensions for the SQL data source, installed for the runtime user.
USER app
RUN python -c "import duckdb; c = duckdb.connect(); c.sql('INSTALL httpfs; INSTALL spatial; INSTALL h3 FROM community')"

ARG APP_VERSION=dev
ENV APP_VERSION=$APP_VERSION PORT=8000
EXPOSE 8000

CMD ["glen-mcp-opt"]

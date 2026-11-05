"""SQL data source: sets and params filled by a DuckDB query, so rows never pass through the LLM."""

from __future__ import annotations

import os

import duckdb

MAX_ROWS = int(os.environ.get("MAX_SQL_ROWS", 500_000))
_CA_BUNDLES = ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
               "/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/ca-bundle.pem", "/etc/ssl/cert.pem")


def _q(v: str) -> str:
    return v.replace("'", "''")


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    for ext in ("httpfs", "h3", "spatial"):
        try:
            con.execute(f"LOAD {ext}")
        except duckdb.Error:
            pass
    endpoint = os.environ.get("S3_DEFAULT_ENDPOINT", "rook-ceph-rgw-nautiluss3.rook")
    use_ssl = os.environ.get("S3_DEFAULT_USE_SSL") or ("false" if endpoint.startswith("rook") else "true")
    url_style = os.environ.get("S3_DEFAULT_URL_STYLE", "path")
    try:
        con.execute(
            f"CREATE SECRET s3 (TYPE S3, KEY_ID '', SECRET '', ENDPOINT '{_q(endpoint)}', "
            f"URL_STYLE '{_q(url_style)}', USE_SSL '{_q(use_ssl)}')"
        )
    except duckdb.Error:
        pass  # httpfs unavailable; local queries still work
    # Remote reads only. The CA bundle stays readable for HTTPS.
    cas = [p for p in _CA_BUNDLES if os.path.exists(p)]
    con.execute(f"SET allowed_paths=[{', '.join(repr(p) for p in cas)}]")
    con.execute("SET allowed_directories=['s3://', 'https://', 'http://']")
    con.execute("SET enable_external_access=false")
    con.execute("SET lock_configuration=true")
    return con


def run_query(sql: str) -> tuple[list[str], list[tuple]]:
    """Run one SELECT. Returns (column names, rows). Raises ValueError with a readable message."""
    con = _connect()
    try:
        stmts = con.extract_statements(sql)
        if len(stmts) != 1 or stmts[0].type != duckdb.StatementType.SELECT:
            raise ValueError("query must be exactly one SELECT statement")
        rel = con.sql(sql)
        cols = rel.columns
        rows = rel.limit(MAX_ROWS + 1).fetchall()
    except duckdb.Error as e:
        raise ValueError(f"SQL error: {str(e).splitlines()[0][:300]}") from None
    finally:
        con.close()
    if len(rows) > MAX_ROWS:
        raise ValueError(f"query returned more than {MAX_ROWS} rows")
    return cols, rows

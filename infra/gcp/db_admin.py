"""One-off owner migration via a loopback Cloud SQL Auth Proxy TLS tunnel; URL arrives on stdin, never argv/logs."""
from __future__ import annotations

import argparse
import re
import sys


def migrate(owner_url: str, iam_user: str) -> None:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    from app.storage import migrations, pg_grants

    if not re.fullmatch(r"[a-z0-9-]+@[a-z0-9-]+\.iam", iam_user):
        raise ValueError("invalid IAM service-account database username")
    url = make_url(owner_url)
    # The driver merges query options over the URL authority, including host,
    # port and dbname. Permit only the reviewed local-proxy TLS setting.
    if (url.drivername != "postgresql+psycopg" or url.host != "127.0.0.1" or url.port != 5433
            or url.database != "veritas" or dict(url.query) != {"sslmode": "disable"}):
        raise ValueError("owner connection must use the local Cloud SQL admin proxy at 127.0.0.1:5433/veritas")
    engine = create_engine(url)
    try:
        migrations.upgrade_to_head(engine)
        with engine.begin() as conn:
            conn.execute(text("DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='veritas_app') "
                              "THEN CREATE ROLE veritas_app NOLOGIN; END IF; END $$"))
            conn.execute(text("REVOKE CREATE ON SCHEMA public FROM PUBLIC"))
            conn.execute(text(f'GRANT veritas_app TO "{iam_user}"'))
        pg_grants.apply(engine, "veritas_app")
    finally:
        engine.dispose()
    print("Schema migrated and non-owner runtime grants applied.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iam-user", required=True)
    args = parser.parse_args()
    try:
        migrate(sys.stdin.readline().strip(), args.iam_user)
    except Exception:
        # SQLAlchemy errors can echo connection details; keep owner credentials out of startup/operator logs.
        raise SystemExit("Database migration failed; verify the owner connection, CA, IAM user and database state.") from None


if __name__ == "__main__":
    main()

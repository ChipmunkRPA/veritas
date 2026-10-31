"""Alembic environment.

Programmatic use (`app.storage.migrations`) passes a live connection in `config.attributes["connection"]`.
The CLI (`alembic upgrade head`) connects to `settings.sqlalchemy_url`. SQLite ALTERs use batch mode, and both ways run with
SQLite's foreign keys unenforced (`migrations.migration`).
"""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from app.core.config import settings
from app.storage.db import Base, engine_url
from app.storage.migrations import migration

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _url() -> str:
    return config.get_main_option("sqlalchemy.url") or settings.sqlalchemy_url


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True, render_as_batch=True,
                      dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def _run(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    engine = create_engine(engine_url(_url()), poolclass=pool.NullPool)
    try:
        with migration(engine) as conn:
            _run(conn)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os

config = context.config


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = os.getenv("APP__DB__SQL_ALCHEMY_CONN")

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection
        )

        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
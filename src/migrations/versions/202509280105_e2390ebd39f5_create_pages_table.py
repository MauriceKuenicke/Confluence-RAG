"""Create Initial Pages Table

Revision ID: e2390ebd39f5
Revises:
Create Date: 2025-09-28 01:05:32.785183

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'e2390ebd39f5'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE confluence_pages (
            page_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            markdown TEXT NOT NULL,
            space_id INTEGER NOT NULL,
            version_number INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            PRIMARY KEY (page_id, version_number)
        )
    """)


def downgrade() -> None:
    op.execute("""
        DROP TABLE confluence_pages CASCADE
    """)

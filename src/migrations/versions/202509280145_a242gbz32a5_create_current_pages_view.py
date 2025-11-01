"""Create Latest Pages View

Revision ID: e2390ebd39f5
Revises:
Create Date: 2025-09-28 01:45:11.284227

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a242gbz32a5'
down_revision: Union[str, None] = 'e2390ebd39f5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        create view current_confluence_pages as (
            WITH latest_pages AS (
                SELECT page_id,
                       title,
                       page_url,
                       space_id,
                       created_at,
                       updated_at,
                       version_number,
                       markdown,
                       ROW_NUMBER() OVER (PARTITION BY page_id ORDER BY version_number DESC) AS rn
                FROM confluence_pages
                )
            SELECT page_id,
                   title,
                   page_url,
                   space_id,
                   created_at,
                   updated_at,
                   version_number,
                   markdown
            FROM latest_pages
            WHERE rn = 1
        )
    """)


def downgrade() -> None:
    op.execute("""
        DROP VIEW current_confluence_pages
    """)


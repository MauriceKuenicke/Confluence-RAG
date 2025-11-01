import os
import random
import time
from datetime import datetime, timedelta
from typing import Any

import html2text
from airflow import DAG
from airflow.decorators import task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from sqlalchemy import create_engine, MetaData, Table, select

from AtlassianAPIWrapper import SpaceMetadata, PageMetadata

default_args = {
    "owner": "RAG",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Confluence Spaces to ingest
spaces_list = [65900]


with DAG(
    "Ingest_Space_Pages",
    dag_display_name="Ingest Space Pages",
    default_args=default_args,
    description="Ingest the content of one or multiple Confluence spaces into our document store.",
    schedule=None,
    catchup=False,
    tags=["RAG"],
) as dag:

    # Task 1: Find all pages in the given list of spaces.
    @task(task_id="Space_Page_Information",
          retries=3,
          retry_delay=timedelta(minutes=1),
          doc="Get the page IDs associated with a specified Confluence space.")
    def get_pages_for_space(space_key: int) -> list[int]:
        """
        Extracts pages for the given space from Confluence.
        Returns a list of page ids for that space.
        """
        print(f"Extracting Page IDs for: {space_key}")
        space = SpaceMetadata(space_key)
        pages_in_space = space.unique_page_ids
        print(f"Pages found in Space: {pages_in_space}")
        return pages_in_space

    # Task 2: To flatten the list of lists into a single list of page IDs
    @task(task_id="Flatten_Page_IDs",
          retries=3,
          doc="Flatten all page IDs into a single list.")
    def flatten_page_ids(pages_per_space: list[list[int]]) -> list[int]:
        """
        Flattens a list of lists of page IDs into a single list of page IDs.
        e.g., [[101, 102], [201, 202], [301]] → [101, 102, 201, 202, 301]
        """
        flattened = [page_id for sublist in pages_per_space for page_id in sublist]
        print(f"Flattened Page IDs: {flattened}")
        return flattened

    # Task 3: Extract and transform the content of each page into Markdown format.
    @task(task_id="Page_Extraction_Transform",
          retries=3,
          retry_delay=timedelta(minutes=1),
          doc="Extract the page content and metadata and transform it into a structured format.")
    def extract_and_transform_page_data(page_id: int) -> dict[str, Any]:
        """Processes one individual page for a given space."""
        print(f"Transforming Page Data for Page ID: {page_id}")

        wait_in_s = random.randint(1, 10)
        print(f"Waiting for {wait_in_s} seconds before extracting page data to not overload the API.")
        time.sleep(wait_in_s)

        page = PageMetadata(page_id)
        print("Page extraction successful. Converting content to Markdown...")
        markdown_converter = html2text.html2text(page.raw_content)
        if markdown_converter.replace("\n", "") == "":
            return None
        page_data = {"page_id": page.page_id,
                     "title": page.title,
                     "space_id": page.space_id,
                     "created_at": page.created_at,
                     "updated_at": page.updated_at,
                     "version_number": page.version_number,
                     "url": page.page_url,
                     "markdown": markdown_converter}
        return page_data

    # Task 4: Load processed data into Postgres
    @task(task_id="Selective_Load_To_Postgres",
          retries=3,
          retry_delay=timedelta(minutes=1),
          doc="Load only updated or new pages into Postgres.")
    def load_pages_to_postgres(processed_pages: list[dict[str, Any]]):
        """
        Loads pages into the PostgreSQL database only if their `version_number`
        is different from the existing rows in the database.
        """
        connection_string = os.getenv("APP__DB__SQL_ALCHEMY_CONN")
        if not connection_string:
            raise ValueError("Database connection string (APP__DB__SQL_ALCHEMY_CONN) is not set!")

        engine = create_engine(connection_string)
        table_name = "confluence_pages"
        with engine.begin() as connection:
            metadata = MetaData()
            table = Table(table_name, metadata, autoload_with=engine)

            # Step 1: Fetch existing page IDs and version numbers from the database
            print("Fetching existing page IDs and version numbers...")
            existing_rows = connection.execute(select(table.c.page_id, table.c.version_number)).fetchall()
            existing_data = {row[0]: row[1] for row in existing_rows}  # {page_id: version_number}

            # Step 2: Filter processed_pages based on `version_number`
            pages_to_insert = []
            for page in processed_pages:
                page_id = page["page_id"]
                version_number = page["version_number"]

                # Only include the page if it is new or has a newer version
                if page_id not in existing_data or version_number > existing_data[page_id]:
                    pages_to_insert.append({
                        "page_id": page["page_id"],
                        "title": page["title"],
                        "space_id": page["space_id"],
                        "created_at": page["created_at"],
                        "updated_at": page["updated_at"],
                        "version_number": page["version_number"],
                        "page_url": page["url"],
                        "markdown": page['markdown'],
                    })

            # Step 3: Insert only the new or updated pages
            if pages_to_insert:
                print(f"Inserting {len(pages_to_insert)} updated/new pages into the database...")
                connection.execute(table.insert(), pages_to_insert)
            else:
                print("No new or updated pages to insert.")

            # Step 4: Delete removed pages
            removed_page_ids = set(existing_data.keys()) - set(page["page_id"] for page in processed_pages)
            if removed_page_ids:
                print(f"Deleting {len(removed_page_ids)} removed pages from the database: {removed_page_ids}")
                connection.execute(table.delete().where(table.c.page_id.in_(removed_page_ids)))


    # Task dependencies using dynamic task mapping
    list_of_page_ids_per_space = get_pages_for_space.expand(space_key=spaces_list)
    flattened_page_ids = flatten_page_ids(list_of_page_ids_per_space)
    transformed_data = extract_and_transform_page_data.expand(page_id=flattened_page_ids)
    load_pages_to_postgres_task = load_pages_to_postgres(processed_pages=transformed_data)

    trigger_next_dag = TriggerDagRunOperator(task_id="Trigger_Indexing",
                                             trigger_dag_id="Index_Confluence_Content",
                                             conf={"message": "Triggering from Ingest DAG"},
                                             wait_for_completion=False)
    load_pages_to_postgres_task >> trigger_next_dag

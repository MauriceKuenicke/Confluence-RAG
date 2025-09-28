from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
from app.qdrant import QdrantConfluencePages
import os
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from sqlalchemy import create_engine
from uuid import uuid4
from langchain_core.documents import Document

default_args = {
    "owner": "RAG",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
        "Index_Confluence_Content",
        dag_display_name="Index Confluence Content",
        default_args=default_args,
        description="Index the content of our document store.",
        schedule=None,
        catchup=False,
        tags=["RAG"],
) as dag:

    # Task 1: Find all pages in the given list of spaces.
    @task(task_id="Index_Page_Content",
          retries=3,
          retry_delay=timedelta(minutes=1),
          doc="Index the content of our document store.")
    def index_content():
        connection_string = os.getenv("APP__DB__SQL_ALCHEMY_CONN")
        if not connection_string:
            raise ValueError("Database connection string (APP__DB__SQL_ALCHEMY_CONN) is not set!")

        engine = create_engine(connection_string)
        with engine.begin() as connection:
            res = connection.execute(
                """SELECT page_id, version_number, title, markdown FROM current_confluence_pages"""
            ).fetchall()  # [(page_id, version_number, title, markdown), ...]

        vector_store = QdrantConfluencePages()
        available_points = vector_store.list_collection_points()


        ###################################################################################################
        # Step 1: Convert data to dictionaries for faster lookups
        res_dict = {page_id: {"version_number": version_number, "title": title, "markdown": markdown} for
                    page_id, version_number, title, markdown in res}
        available_points_dict = {point["page_id"]: point for point in available_points}

        # Step 2: Initialize lists
        to_be_deleted = []  # Point
        to_be_added = []  # Page
        to_be_updated = []  # (Page, Point)

        # Step 3: Determine items to be updated or added
        for page_id, page_data in res_dict.items():
            available_point = available_points_dict.get(page_id)
            if available_point:
                # Check if version numbers are different for updates
                if available_point["version_number"] != page_data["version_number"]:
                    to_be_updated.append((page_id, available_point["point_id"]))
            else:
                # If page_id is not in available points, it needs to be added
                to_be_added.append(page_id)

        # Step 4: Determine items to be deleted
        for point in available_points:
            if point["page_id"] not in res_dict:
                to_be_deleted.append(point["point_id"])
        ###################################################################################################

        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        qdrant = QdrantVectorStore(
            client=vector_store.client,
            collection_name=os.getenv("COLLECTION_NAME"),
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.SPARSE,
            sparse_vector_name="sparse_content",
        )
        # UPDATE EXISTING
        if to_be_updated:
            for doc_id, point_id in to_be_updated:
                print(f"Updating point {point_id} with new version for page: {doc_id}.")
                qdrant.delete(ids=[point_id])
                qdrant.add_documents(
                    documents=[Document(page_content=res_dict[doc_id]["markdown"],
                                        metadata={"page_id": doc_id,
                                                  "version_number": res_dict[doc_id]["version_number"],
                                                  "title": res_dict[doc_id]["title"]})],
                    ids=[str(uuid4())],
                )

        # DELETE REMOVED
        if to_be_deleted:
            print(f"Deleting points: {to_be_deleted}.")
            qdrant.delete(ids=to_be_deleted)

        # ADD NEW
        if to_be_added:
            for doc_id in to_be_added:
                print(f"Adding new page: {doc_id} to the database.")
                qdrant.add_documents(
                    documents=[Document(page_content=res_dict[doc_id]["markdown"],
                                        metadata={"page_id": doc_id,
                                                  "version_number": res_dict[doc_id]["version_number"],
                                                  "title": res_dict[doc_id]["title"]})],
                    ids=[str(uuid4())],
                )

        print(f"Page IDs added: {to_be_added}")
        print(f"Existing entries updated (Page, Point): {to_be_updated}")
        print(f"Points deleted: {to_be_deleted}")

        return None

    index_content()
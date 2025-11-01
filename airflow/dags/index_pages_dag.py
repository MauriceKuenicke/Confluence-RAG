import os
from datetime import datetime, timedelta
from uuid import uuid4

from airflow import DAG
from airflow.decorators import task
from app.qdrant import QdrantConfluencePages
from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
from qdrant_client.models import PointStruct
from sqlalchemy import create_engine


def chunk_markdown_content(
        content: str,
        chunk_size: int = 1000,
        overlap_size: int = 200,
        min_chunk_size: int = 100
) -> list[dict]:
    """
    Chunk markdown content into overlapping segments for better retrieval.

    Args:
        content: The raw Markdown content to chunk
        chunk_size: Maximum size of each chunk in characters (default: 1000)
        overlap_size: Number of characters to overlap between chunks (default: 200)
        min_chunk_size: Minimum size for a chunk to be included (default: 100)
        split_on_headers: Try to split on Markdown headers when possible (default: True)
        preserve_code_blocks: Avoid splitting inside code blocks (default: True)

    Returns:
        List of dictionaries with keys:
          - 'text': chunk content with line breaks collapsed to single spaces
          - 'start_line': 1-based start line number in the original content
          - 'end_line': 1-based end line number in the original content
    """
    if not content or len(content.strip()) == 0:
        return []

    # Prepare for line-based accounting
    lines = content.split('\n')
    total_lines = len(lines)

    # If content is smaller than chunk_size, return as a single chunk
    if len(content) <= chunk_size:
        return [{
            "text": " ".join(content.split()),
            "start_line": 1,
            "end_line": total_lines
        }]

    # Group content into paragraphs (sequences of non-empty lines), tracking line numbers
    paragraphs: list[dict] = []
    current_para_lines: list[str] = []
    current_start_line: int | None = None

    for idx, line in enumerate(lines, start=1):
        if line.strip() == "":
            if current_para_lines:
                para_text = "\n".join(current_para_lines)
                para_end_line = (current_start_line or idx) + len(current_para_lines) - 1
                paragraphs.append({
                    "text": para_text,
                    "start_line": current_start_line,
                    "end_line": para_end_line,
                })
                current_para_lines = []
                current_start_line = None
            # consecutive blank lines just separate paragraphs; no-op
        else:
            if current_start_line is None:
                current_start_line = idx
            current_para_lines.append(line)

    # Finalize last paragraph (if any)
    if current_para_lines:
        para_text = "\n".join(current_para_lines)
        para_end_line = (current_start_line or total_lines) + len(current_para_lines) - 1
        paragraphs.append({
            "text": para_text,
            "start_line": current_start_line,
            "end_line": para_end_line,
        })

    # If still no paragraphs (e.g., all blank), return empty
    if not paragraphs:
        return []

    chunks: list[dict] = []

    current_chunk_text: str = ""
    chunk_start_line: int | None = None
    chunk_end_line: int | None = None

    for para in paragraphs:
        para_text = para["text"]
        # length if we add this paragraph (accounting for the "\n\n" between paragraphs)
        join_text = ("\n\n" if current_chunk_text else "") + para_text
        would_exceed = len(current_chunk_text) + len(join_text) > chunk_size

        if would_exceed and current_chunk_text:
            # finalize current chunk
            collapsed = " ".join(current_chunk_text.strip().split())
            if len(collapsed) >= min_chunk_size:
                chunks.append({
                    "text": collapsed,
                    "start_line": chunk_start_line or para["start_line"],
                    "end_line": chunk_end_line or para["end_line"],
                })

            # start next chunk with overlap
            if overlap_size > 0 and len(current_chunk_text) > overlap_size:
                overlap_text = current_chunk_text[-overlap_size:]
                # Try to start overlap at a word boundary
                space_idx = overlap_text.find(" ")
                if space_idx > 0:
                    overlap_text = overlap_text[space_idx + 1:]

                # Estimate new start line by counting newlines in the overlap
                overlap_line_count = overlap_text.count("\n")
                new_start_line = max(1, (chunk_end_line or para["end_line"]) - overlap_line_count)

                current_chunk_text = overlap_text + "\n\n" + para_text
                chunk_start_line = new_start_line
                chunk_end_line = para["end_line"]
            else:
                # No overlap: start fresh from this paragraph
                current_chunk_text = para_text
                chunk_start_line = para["start_line"]
                chunk_end_line = para["end_line"]
        else:
            # add paragraph to current chunk
            current_chunk_text = current_chunk_text + join_text if current_chunk_text else para_text
            chunk_end_line = para["end_line"]
            if chunk_start_line is None:
                chunk_start_line = para["start_line"]

    # finalize trailing chunk
    if current_chunk_text:
        collapsed = " ".join(current_chunk_text.strip().split())
        if len(collapsed) >= min_chunk_size:
            chunks.append({
                "text": collapsed,
                "start_line": chunk_start_line or 1,
                "end_line": chunk_end_line or total_lines,
            })

    return chunks


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
                """SELECT page_id, version_number, title, markdown, page_url
                   FROM current_confluence_pages"""
            ).fetchall()  # [(page_id, version_number, title, markdown), ...]

        vector_store = QdrantConfluencePages()
        available_points = vector_store.list_collection_points()


        ###################################################################################################
        # Step 1: Convert data to dictionaries for faster lookups
        res_dict = {
            page_id: {"version_number": version_number, "title": title, "markdown": markdown, "page_url": page_url} for
            page_id, version_number, title, markdown, page_url in res}
        available_points_dict = {}
        for p in available_points:
            id_page = p["page_id"]
            if id_page not in available_points_dict:
                available_points_dict[id_page] = []
            available_points_dict[id_page].append({
                "point_id": p["point_id"],
                "version_number": p["version_number"]
            })

        # Step 2: Initialize lists
        to_be_deleted = []
        to_be_added = []
        to_be_updated = []

        # Step 3: Determine items to be updated or added
        for page_id, page_data in res_dict.items():
            points_for_page = available_points_dict.get(page_id)
            if points_for_page:
                # All points should have the same version number
                if points_for_page[0]["version_number"] != page_data["version_number"]:
                    to_be_updated.append(page_id)
            else:
                # If page_id is not in available points, it needs to be added
                to_be_added.append(page_id)

        # Step 4: Determine items to be deleted
        for point in available_points:
            if point["page_id"] not in res_dict:
                to_be_deleted.append(point["page_id"])
        ###################################################################################################
        # DELETE REMOVED
        if to_be_deleted:
            for page_id_to_delete in to_be_deleted:
                delete_filter = Filter(
                    must=[
                        FieldCondition(
                            key="page_id",
                            match=MatchValue(value=page_id_to_delete)
                        )
                    ]
                )
                points_selector = FilterSelector(filter=delete_filter)
                vector_store.client.delete(collection_name=vector_store.collection, points_selector=points_selector)

        # UPDATE EXISTING
        if to_be_updated:
            for page_id_to_update in to_be_updated:
                delete_filter = Filter(
                    must=[
                        FieldCondition(
                            key="page_id",
                            match=MatchValue(value=page_id_to_update)
                        )
                    ]
                )
                points_selector = FilterSelector(filter=delete_filter)
                vector_store.client.delete(collection_name=vector_store.collection, points_selector=points_selector)
                to_be_added.append(page_id_to_update)


        # ADD NEW
        if to_be_added:
            for doc_id in to_be_added:
                print(f"Adding new page: {doc_id} to the database.")
                doc_content = res_dict[doc_id]["markdown"]
                chunks = chunk_markdown_content(doc_content)
                chunk_id = 0
                for c in chunks:
                    sparse_content_vec = list(vector_store.sparse_embedding_model.embed([c["text"]]))[0]
                    sparse_title_vec = list(vector_store.sparse_embedding_model.embed([res_dict[doc_id]["title"]]))[0]
                    dense_content_vec = list(vector_store.dense_embedding_model.embed([c["text"]]))[0]
                    metadata = {"page_id": doc_id,
                                "version_number": res_dict[doc_id]["version_number"],
                                "page_title": res_dict[doc_id]["title"],
                                "page_url": res_dict[doc_id]["page_url"],
                                "chunk_seq": chunk_id,
                                "total_chunks": len(chunks),
                                "chunk_content": c["text"],
                                "chunk_size": len(c["text"]),
                                "loaded_at": datetime.now().isoformat(),
                                "start_line": c["start_line"],
                                "end_line": c["end_line"]}

                    point = PointStruct(
                        id=str(uuid4()),
                        vector={  # type: ignore
                            "dense_content": dense_content_vec,
                            "sparse_content": sparse_content_vec.as_object(),  # type: ignore
                            "sparse_title": sparse_title_vec.as_object(),  # type: ignore
                        },
                        payload=metadata,
                    )

                    vector_store.client.upsert(collection_name=vector_store.collection, points=[point])
                    chunk_id += 1

        print(f"Page IDs added/updated: {to_be_added}")
        print(f"Existing entries updated: {to_be_updated}")
        print(f"Pages deleted: {to_be_deleted}")

        return None

    index_content()
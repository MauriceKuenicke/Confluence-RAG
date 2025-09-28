import os
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
import logging

logger = logging.getLogger('uvicorn.error')

class QdrantConfluencePages:
    def __init__(self) -> None:
        self._client = QdrantClient(url=os.getenv("APP__VEC__QDRANT_URL"),
                                    api_key=os.getenv("APP__VEC__QDRANT_API_KEY"))
        self._collection = os.getenv("COLLECTION_NAME")

    @property
    def client(self) -> QdrantClient:
        return self._client


    def create_collection(self, force: bool = False) -> None:
        logger.info(f"Checking if {self._collection} exists.")
        exists = self._client.collection_exists(self._collection)
        if exists and not force:
            logger.info(f"Collection {self._collection} already exists.")
            return

        if exists and force:
            logger.info(f"Collection {self._collection} already exists. Deleting and re-creating.")
            self._client.delete_collection(self._collection)

        self._client.create_collection(collection_name=self._collection,
                                       vectors_config={},
                                       sparse_vectors_config={
                                           "sparse_content": models.SparseVectorParams(
                                               index=models.SparseIndexParams(on_disk=False),
                                               modifier=models.Modifier.IDF,  # **required** for BM25/BM42
                                           )
                                       })
        logger.info(f"Collection {self._collection} created.")

    def list_collection_points(self):
        r = self._client.query_points(collection_name=self._collection, limit=10000000000, with_payload=["metadata"])
        res = []
        for x in r.points:
            res.append({
                "point_id": x.id,
                "page_id": x.payload["metadata"]["page_id"],
                "version_number": x.payload["metadata"]["version_number"]
            })
        return res

import logging
import os

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, models

logger = logging.getLogger('uvicorn.error')


class QdrantConfluencePages:
    def __init__(self) -> None:
        self._client = QdrantClient(url=os.getenv("APP__VEC__QDRANT_URL"),
                                    api_key=os.getenv("APP__VEC__QDRANT_API_KEY"))
        self._collection = os.getenv("COLLECTION_NAME")
        self._dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        self._sparse_embedding_model = SparseTextEmbedding("Qdrant/bm25")

    @property
    def client(self) -> QdrantClient:
        return self._client

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def dense_embedding_model(self) -> TextEmbedding:
        return self._dense_embedding_model

    @property
    def sparse_embedding_model(self) -> SparseTextEmbedding:
        return self._sparse_embedding_model


    def create_collection(self, force: bool = False) -> None:
        logger.info(f"Checking if {self._collection} exists.")
        exists = self._client.collection_exists(self._collection)
        if exists and not force:
            logger.info(f"Collection {self._collection} already exists.")
            return

        if exists and force:
            logger.info(f"Collection {self._collection} already exists. Deleting and re-creating.")
            self._client.delete_collection(self._collection)

        test_embedding = list(self._dense_embedding_model.embed("Hello World"))[0]
        self._client.create_collection(collection_name=self._collection,
                                       vectors_config={
                                           "dense_content": models.VectorParams(size=len(test_embedding),
                                                                                distance=Distance.COSINE),
                                       },
                                       sparse_vectors_config={
                                           "sparse_content": models.SparseVectorParams(
                                               index=models.SparseIndexParams(on_disk=False),
                                               modifier=models.Modifier.IDF,  # **required** for BM25/BM42
                                           ),
                                           "sparse_title": models.SparseVectorParams(
                                               index=models.SparseIndexParams(on_disk=False),
                                               modifier=models.Modifier.IDF,
                                           ),
                                       })
        logger.info(f"Collection {self._collection} created.")

    def list_collection_points(self):
        r = self._client.query_points(collection_name=self._collection, limit=10000000000,
                                      with_payload=["page_id", "version_number"])
        res = []
        for x in r.points:
            res.append({
                "point_id": x.id,
                "page_id": x.payload["page_id"],
                "version_number": x.payload["version_number"]
            })
        return res

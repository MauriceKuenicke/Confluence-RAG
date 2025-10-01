from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import alembic.config
from .qdrant import QdrantConfluencePages
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
import sys
import logging
import os
from fastembed.rerank.cross_encoder import TextCrossEncoder

encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
reranker = TextCrossEncoder(model_name='jinaai/jina-reranker-v2-base-multilingual')


logger = logging.getLogger('uvicorn.error')

app = FastAPI(title="RAG Backend API",
              swagger_ui_parameters={"defaultModelsExpandDepth": -1},
              docs_url="/api/docs",
              )

app.add_middleware(
    CORSMiddleware,  # NOQA
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthcheck")
def healthcheck():
    return {"Status": "Everything OK."}

@app.post("/search")
def search(query: str):
    vector_store = QdrantConfluencePages()
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    qdrant = QdrantVectorStore(
        client=vector_store.client,
        collection_name=os.getenv("COLLECTION_NAME"),
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.SPARSE,
        sparse_vector_name="sparse_content",
    )

    found_docs = qdrant.similarity_search(query, k=5)

    description_hits = [x.page_content for x in found_docs]
    new_scores = list(reranker.rerank(query, description_hits))
    ranking = [(i, score) for i, score in enumerate(new_scores)]
    ranking.sort(key=lambda x: x[1], reverse=True)
    found_docs = [found_docs[i] for i, _ in ranking]
    return found_docs

if "pytest" not in sys.modules:
    alembic.config.main(argv=["--raiseerr", "upgrade", "head"])
    qdrant_wrapper = QdrantConfluencePages()
    logger.info("Creating Qdrant collection if not exist")
    qdrant_wrapper.create_collection()

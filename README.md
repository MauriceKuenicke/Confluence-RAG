# Confluence RAG

## High-level Architecture
This project ingests Confluence content, indexes it for retrieval, and exposes search via a simple backend API. At a glance:
- Airflow orchestrates two pipelines:
  - Ingest: discover pages in specified Confluence spaces, extract content, convert to Markdown, and selectively load into the application database based on version changes.
  - Index: read the current pages from the application DB and keep the Qdrant vector store in sync (add/update/delete) based on page versions.
- Two Postgres instances are used: one for Airflow's own metadata and one for the application data (ingested pages).
- A FastAPI backend provides a thin API layer and exposes a /search endpoint that queries Qdrant via LangChain using sparse BM25 retrieval. Alembic ensures DB schema migrations run on startup.
- Everything is containerized and wired together via Docker Compose.

```mermaid
flowchart LR
  A[Confluence Cloud] -->|REST API| B[Airflow DAG: Ingest_Space_Pages]
  B --> B1[Get space pages]
  B1 --> B2[Extract & transform to Markdown]
  B2 -->|Selective Load using version check| D[(Application Postgres)]
  B -->|trigger| C[Airflow DAG: Index_Confluence_Content]
  C -->|Sync add/update/delete| Q[Qdrant Vector DB]
  D -->|Load Current Documents| C

  subgraph Airflow Runtime
    direction TB
    B
    B1
    B2
    C
  end

  B --> E[(Airflow Postgres)]

  F[FastAPI Backend API] -->|/search via LangChain BM25 sparse + Re-Ranking| Q
  U[Users / Integrations] -->|HTTP| F
```

Key data flow:
- Confluence -> Airflow (Ingest): Page metadata and HTML content fetched via REST.
- Airflow (Ingest) -> App DB: Only new/updated pages inserted (checked by page version).
- App DB -> Airflow (Index) -> Qdrant: Indexing keeps Qdrant synchronized (adds, updates, deletes) by version.
- FastAPI -> Qdrant: /search queries Qdrant using sparse BM25 retrieval; responses returned to clients.
- Backend API -> Clients: Read-only access and healthchecks.

## Services Description & Availability

| **Service**                       | **Port**       |
|-----------------------------------|----------------|
| Airflow UI                        | 8080           |
| Airflow Backend DB (Postgres)     | Not Exposed    |
| Application Backend DB (Postgres) | 5432           |
| Application API (FastAPI)         | 8005/api/docs  |
| Qdrant Vector DB                  | 6333/dashboard |


The Airflow logs as well as the Postgres database data are managed using named
volumes to persist them across restarts and upgrades.
Named volumes are better for runtime-generated data,
where you do not need direct access to raw files on the host.
They abstract the data storage away from the file system and
provide more portability (e.g., moving services between hosts).

## Setup
We use a single, global, environment variables file called `.env` which should be located on the root-level of 
the project. 
A template file `.env.template` is provided to showcase the expected variables which should be provided. With the
environment variables set up, you can run
```sh
docker-compose up
```
to start-up the container.

### Initial Airflow User
On first start, a default Airflow user is created with username `admin` and password `admin`.

### Backend API Migrations
Migrations are automatically executed on start-up of the backend service. You can
manually execute migrations inside the docker container by running
```sh
docker exec -it <container_name> alembic <command>
```

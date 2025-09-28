from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import alembic.config
import sys


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

if "pytest" not in sys.modules:
    alembic.config.main(argv=["--raiseerr", "upgrade", "head"])
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import load_env_file

load_env_file()

from .indexer import load_vector_index
from .routes import router
from .chat_routes import router as chat_router
from .conversations import get_repository

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def get_allowed_origins() -> list[str]:
    raw_origins = os.getenv("ALLOWED_ORIGINS")
    if not raw_origins:
        return DEFAULT_ALLOWED_ORIGINS
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


app = FastAPI(title="Vector RAG Knowledge Base Q&A Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(chat_router)


@app.on_event("startup")
def load_persisted_index():
    # Fail startup if a saved index cannot be loaded; accepting writes would
    # otherwise overwrite the registry of previously uploaded documents.
    load_vector_index()
    get_repository()

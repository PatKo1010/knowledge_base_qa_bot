import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import load_env_file

load_env_file()

from .indexer import load_vector_index
from .routes import router

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


@app.on_event("startup")
def load_persisted_index():
    try:
        load_vector_index()
    except Exception as exc:
        print(f"[vector_rag] Skipping persisted FAISS load: {exc}", flush=True)

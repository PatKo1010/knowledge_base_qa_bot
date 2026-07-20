from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import load_env_file

load_env_file()

from .indexer import load_vector_index
from .routes import router

app = FastAPI(title="Vector RAG Knowledge Base Q&A Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
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

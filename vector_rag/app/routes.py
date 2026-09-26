from fastapi import APIRouter, UploadFile
from starlette.concurrency import run_in_threadpool
import logging
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from .indexer import build_index, chunk_manifest, ingest_pdf
from .pdf_ingestion import MAX_UPLOAD_BYTES
from .retrieval import query, stream_query
from .schemas import ChatRequest, ChatResponse, IndexResponse, UploadResponse

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/chunks")
def chunks():
    return {"chunks": chunk_manifest()}


@router.post("/index", response_model=IndexResponse)
def index_docs():
    try:
        files_count, sections_count = build_index()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IndexResponse(files_indexed=files_count, sections_indexed=sections_count)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return query(req.query)


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    return StreamingResponse(
        stream_query(req.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile):
    try:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="PDF must be 20 MB or smaller.")
        try:
            return await run_in_threadpool(ingest_pdf, data, file.filename or "document.pdf")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logging.getLogger(__name__).exception("PDF ingestion failed")
            raise HTTPException(status_code=500, detail="PDF indexing failed. Please retry.") from exc
    finally:
        await file.close()

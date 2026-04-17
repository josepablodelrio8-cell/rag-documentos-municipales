"""
main.py — API REST del sistema RAG.
Endpoints:
  POST /upload  → sube y procesa un PDF
  POST /query   → hace una pregunta sobre los documentos cargados
  GET  /docs    → lista documentos cargados
  GET  /        → health check
"""
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import ingest_pdf, query, get_loaded_documents

# ------------------------------------------------------------------ #
# App                                                                  #
# ------------------------------------------------------------------ #
app = FastAPI(
    title="RAG — Documentos Municipales",
    description="API para consultar documentos municipales en lenguaje natural.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # en producción, limitar al dominio del frontend
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------ #
# Schemas                                                              #
# ------------------------------------------------------------------ #
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


# ------------------------------------------------------------------ #
# Endpoints                                                            #
# ------------------------------------------------------------------ #
@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG API activa", "swagger": "/docs"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Sube un PDF y lo procesa para que pueda ser consultado."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF.")

    dest = DATA_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        n_chunks = ingest_pdf(str(dest), file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "message": f"Documento '{file.filename}' procesado correctamente.",
        "chunks_generados": n_chunks,
    }


@app.post("/query", response_model=QueryResponse)
async def ask(req: QueryRequest):
    """Hace una pregunta sobre los documentos cargados."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    result = query(req.question)
    return result


@app.get("/documents")
def list_documents():
    """Lista los documentos actualmente cargados en memoria."""
    docs = get_loaded_documents()
    return {"documentos_cargados": docs, "total": len(docs)}
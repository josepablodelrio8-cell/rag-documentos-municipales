"""
rag_engine.py — Núcleo del sistema RAG.
Maneja la ingesta de PDFs, generación de embeddings,
índice FAISS y consulta con LLM.
"""
import os
import faiss
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------ #
# Configuración                                                        #
# ------------------------------------------------------------------ #
CHUNK_SIZE = 512        # palabras por fragmento
OVERLAP = 50            # palabras de solapamiento entre fragmentos
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Modelo de embeddings (se descarga automáticamente la primera vez)
embedder = SentenceTransformer(MODEL_NAME)

# Estado en memoria
chunks_store: list[dict] = []   # [{"text": "...", "source": "archivo.pdf"}]
faiss_index = None               # índice FAISS


# ------------------------------------------------------------------ #
# Utilidades de texto                                                  #
# ------------------------------------------------------------------ #

def _split_text(text: str) -> list[str]:
    """Divide el texto en fragmentos con solapamiento."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ------------------------------------------------------------------ #
# Ingesta de documentos                                                #
# ------------------------------------------------------------------ #

def ingest_pdf(pdf_path: str, filename: str) -> int:
    """
    Procesa un PDF y lo agrega al índice FAISS.
    Retorna la cantidad de fragmentos generados.
    """
    global faiss_index, chunks_store

    # 1. Extraer texto del PDF
    reader = PdfReader(pdf_path)
    full_text = "\n".join(
        page.extract_text() or "" for page in reader.pages
    )

    if not full_text.strip():
        raise ValueError(f"No se pudo extraer texto de '{filename}'. ¿Es un PDF escaneado?")

    # 2. Dividir en fragmentos
    chunks = _split_text(full_text)

    # 3. Generar embeddings
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")

    # 4. Agregar al índice FAISS
    if faiss_index is None:
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dim)

    faiss_index.add(embeddings)

    # 5. Guardar chunks con su fuente
    for chunk in chunks:
        chunks_store.append({"text": chunk, "source": filename})

    return len(chunks)


def get_loaded_documents() -> list[str]:
    """Retorna la lista de documentos actualmente cargados."""
    return list({c["source"] for c in chunks_store})


# ------------------------------------------------------------------ #
# Consulta                                                             #
# ------------------------------------------------------------------ #

def query(question: str, top_k: int = 4) -> dict:
    """
    Busca los fragmentos más relevantes y genera una respuesta con el LLM.
    Retorna: {"answer": str, "sources": list[str]}
    """
    if faiss_index is None or not chunks_store:
        return {
            "answer": "No hay documentos cargados. Por favor sube un PDF primero.",
            "sources": []
        }

    # 1. Embedding de la pregunta
    q_embedding = embedder.encode([question], show_progress_bar=False)
    q_embedding = np.array(q_embedding, dtype="float32")

    # 2. Búsqueda por similitud en FAISS
    distances, indices = faiss_index.search(q_embedding, top_k)
    relevant_chunks = [
        chunks_store[i] for i in indices[0] if i < len(chunks_store)
    ]

    # 3. Construir contexto y llamar al LLM
    context = "\n\n---\n\n".join(c["text"] for c in relevant_chunks)
    sources = list({c["source"] for c in relevant_chunks})
    answer = _call_llm(question, context)

    return {"answer": answer, "sources": sources}


def _call_llm(question: str, context: str) -> str:
    """Llama al LLM con la pregunta y el contexto recuperado."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = """Eres un asistente especializado en documentos municipales peruanos.
Responde ÚNICAMENTE basándote en el contexto proporcionado.
Si la respuesta no está en el contexto, di exactamente: "No encontré información sobre eso en los documentos cargados."
Sé preciso, cita artículos o numerales cuando los encuentres.
Responde en español."""

    user_prompt = f"""CONTEXTO DE LOS DOCUMENTOS:
{context}

PREGUNTA: {question}

RESPUESTA:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=600,
        temperature=0.1,    # baja temperatura = respuestas más precisas y consistentes
    )

    return response.choices[0].message.content.strip()
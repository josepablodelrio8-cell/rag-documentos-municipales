# 🏛️ RAG — Sistema de Consulta de Documentos Municipales

> Chatbot de inteligencia artificial que permite hacer preguntas en lenguaje natural sobre documentos oficiales (ordenanzas, resoluciones, directivas) y obtener respuestas citando la fuente exacta.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-1.8-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## ¿Qué problema resuelve?

En instituciones públicas, el personal dedica horas a revisar manualmente cientos de documentos (ordenanzas, resoluciones, circulares del MEF, normativas SIAF) para responder consultas internas. Este sistema permite hacer preguntas como:

- *"¿Cuál es la tasa de arbitrios para predios comerciales según la Ordenanza N°045?"*
- *"¿Qué documentos se requieren para solicitar una licencia de funcionamiento?"*
- *"¿Qué dice la directiva sobre el cierre del ejercicio presupuestal?"*

Y obtener una respuesta inmediata, fundamentada en el documento real, con la fuente citada.

---

## Demo

```
Usuario: ¿Cuáles son los requisitos para obtener una licencia de funcionamiento?

Asistente: Según la Ordenanza Municipal N°023-2023, los requisitos son:
1. Solicitud dirigida al alcalde
2. Copia del DNI del titular o representante legal
3. Vigencia de poder (personas jurídicas)
4. Declaración jurada de observancia de condiciones de seguridad

📎 Fuentes: ordenanza_023_2023.pdf
```

---

## Arquitectura

```
PDFs ──► Text Splitter ──► Embeddings (sentence-transformers)
                                  │
                             FAISS Index
                                  │
Pregunta ──► Embedding ──► Búsqueda similitud ──► Top-K fragmentos
                                                        │
                                               Prompt + contexto
                                                        │
                                                LLM (GPT-4o-mini)
                                                        │
                                           Respuesta + fuente citada
```

**Stack:**

| Componente | Tecnología |
|---|---|
| Backend API | FastAPI |
| Embeddings | sentence-transformers (multilingual) |
| Vector Store | FAISS (Facebook AI) |
| LLM | OpenAI GPT-4o-mini |
| Frontend | Streamlit |
| Extracción PDF | PyPDF |

---

## Instalación

### Requisitos
- Python 3.11+
- API key de OpenAI o Groq (Groq tiene tier gratuito)

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/rag-documentos-municipales.git
cd rag-documentos-municipales
```

### 2. Instalar dependencias del backend
```bash
cd backend
pip install -r requirements.txt
```

### 3. Configurar variables de entorno
```bash
cp .env.example .env
# Editar .env y poner tu API key real
```

### 4. Ejecutar el backend
```bash
cd backend
uvicorn main:app --reload
# API disponible en http://localhost:8000
# Documentación Swagger en http://localhost:8000/docs
```

### 5. Ejecutar el frontend (otra terminal)
```bash
cd frontend
pip install streamlit
streamlit run app.py
# Interfaz en http://localhost:8501
```

---

## Uso via API

**Subir un documento:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@ordenanza_045.pdf"
```

**Hacer una pregunta:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuáles son los requisitos para licencia de funcionamiento?"}'
```

**Respuesta:**
```json
{
  "answer": "Según los documentos cargados, los requisitos son...",
  "sources": ["ordenanza_045.pdf"]
}
```

---

## Estructura del proyecto

```
rag-documentos-municipales/
├── backend/
│   ├── main.py          # FastAPI — endpoints REST
│   ├── rag_engine.py    # Lógica de ingesta, embeddings y consulta
│   ├── requirements.txt
│   └── data/            # PDFs cargados (no se suben a Git)
├── frontend/
│   └── app.py           # Interfaz Streamlit con chat
├── .env.example         # Plantilla de variables de entorno
├── .gitignore
└── README.md
```

---

## Posibles extensiones
- [ ] Persistencia del índice FAISS en disco
- [ ] Soporte para Word (.docx) y Excel
- [ ] Historial de conversaciones por sesión
- [ ] Autenticación de usuarios
- [ ] Deploy en Railway o Render con Docker

---

## Contexto

Este proyecto nace de una necesidad real del sector público peruano: acceder rápidamente a información dispersa en documentos oficiales. La técnica RAG permite construir sistemas de consulta sobre documentos privados sin exponer información sensible al LLM, ya que este solo recibe los fragmentos relevantes para cada pregunta.

---

## Autor

**Pablo Jose Del Rio Mayta**
Software Engineer · Municipalidad Provincial de Yauli - La Oroya
[GitHub](https://github.com/TU_USUARIO) · [LinkedIn](https://linkedin.com/in/TU_USUARIO)

---

## Licencia

MIT
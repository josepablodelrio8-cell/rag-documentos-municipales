"""
app.py — Interfaz web del sistema RAG.
Ejecutar con: streamlit run app.py
Requiere que el backend esté corriendo en http://localhost:8000
"""
import streamlit as st
import requests

API_URL = "http://localhost:8000"

# ------------------------------------------------------------------ #
# Configuración de página                                              #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Consulta de Documentos Municipales",
    page_icon="🏛️",
    layout="centered",
)

st.title("🏛️ Consulta de Documentos Municipales")
st.caption("Sistema RAG — Municipalidad Provincial de Yauli · La Oroya")
st.divider()

# ------------------------------------------------------------------ #
# Sidebar — gestión de documentos                                      #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.header("📂 Documentos")

    uploaded_file = st.file_uploader(
        "Sube un PDF (ordenanza, resolución, directiva...)",
        type="pdf",
        help="El documento se procesará y quedará disponible para consultas.",
    )

    if uploaded_file and st.button("Procesar documento", type="primary"):
        with st.spinner("Procesando PDF..."):
            resp = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
            )
        if resp.ok:
            data = resp.json()
            st.success(f"✓ {data['message']}")
            st.info(f"Fragmentos generados: {data['chunks_generados']}")
        else:
            st.error(f"Error: {resp.json().get('detail', 'Error desconocido')}")

    st.divider()
    st.subheader("Documentos cargados")
    try:
        resp = requests.get(f"{API_URL}/documents", timeout=3)
        if resp.ok:
            docs = resp.json()["documentos_cargados"]
            if docs:
                for doc in docs:
                    st.markdown(f"- 📄 {doc}")
            else:
                st.caption("Ninguno todavía.")
    except Exception:
        st.caption("Backend no disponible.")

# ------------------------------------------------------------------ #
# Área principal — chat                                                #
# ------------------------------------------------------------------ #

# Historial de conversación en session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📎 Fuentes: {', '.join(msg['sources'])}")

# Input de pregunta
question = st.chat_input("Escribe tu pregunta sobre los documentos...")

if question:
    # Mostrar pregunta del usuario
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Consultar al backend
    with st.chat_message("assistant"):
        with st.spinner("Buscando en los documentos..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"question": question},
                    timeout=30,
                )
                if resp.ok:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])

                    st.markdown(answer)
                    if sources:
                        st.caption(f"📎 Fuentes: {', '.join(sources)}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                else:
                    err = resp.json().get("detail", "Error desconocido")
                    st.error(f"Error del backend: {err}")
            except requests.exceptions.ConnectionError:
                st.error("No se puede conectar al backend. ¿Está corriendo `uvicorn main:app`?")
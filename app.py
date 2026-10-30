from __future__ import annotations

DB_NAME = "pdf_embeddings_db"
COLLECTION_NAME = "pdf_vectors"
VECTOR_INDEX_NAME = "vector_index"

from dotenv import load_dotenv
import os
import streamlit as st
from pymongo import MongoClient
from google import genai

from rag_core import answer_question

st.set_page_config(page_title='Chatbot RAG - AWS Cloud Adoption Framework', page_icon='☁️', layout="wide")
st.title('☁️' + " " + 'Chatbot RAG - AWS Cloud Adoption Framework')
st.caption('Respuestas fundamentadas en AWS Cloud Adoption Framework (PDF de Canvas).')

load_dotenv()

def read_secret(name: str) -> str | None:
    """Read from Streamlit Cloud [app] secrets first, then local environment."""
    try:
        value = st.secrets["app"][name]
        if value:
            return str(value)
    except Exception:
        pass
    value = os.getenv(name)
    return value if value else None


GOOGLE_API_KEY = read_secret("GOOGLE_API_KEY")
MONGODB_URI = read_secret("MONGODB_URI")
GENERATION_MODEL = read_secret("GENERATION_MODEL") or "gemini-3.5-flash"

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("Faltan credenciales. Configura GOOGLE_API_KEY y MONGODB_URI en .streamlit/secrets.toml o en Streamlit Cloud → Settings → Secrets.")
    st.code('[app]\nGOOGLE_API_KEY = "TU_NUEVA_API_KEY"\nMONGODB_URI = "mongodb+srv://USUARIO:CONTRASENA@CLUSTER.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"', language="toml")
    st.stop()


@st.cache_resource(show_spinner="Conectando con Gemini y MongoDB Atlas...")
def get_resources(api_key: str, mongodb_uri: str):
    genai_client = genai.Client(api_key=api_key)
    mongo_client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=10000)
    mongo_client.admin.command("ping")
    collection = mongo_client[DB_NAME][COLLECTION_NAME]
    return genai_client, mongo_client, collection


try:
    client_genai, client_mongo, collection = get_resources(GOOGLE_API_KEY, MONGODB_URI)
    total_docs = collection.count_documents({})
except Exception as exc:
    st.error("No se pudo conectar con los servicios. Revisa la URI, el usuario de base de datos y Network Access (IP).")
    st.caption(f"Tipo de error: {type(exc).__name__}")
    st.stop()

with st.sidebar:
    st.subheader("Configuración")
    k = st.selectbox("Fragmentos recuperados (k)", options=[3, 5, 10], index=1)
    st.metric("Fragmentos en la colección", f"{total_docs:,}")
    st.caption(f"Base: `{DB_NAME}`")
    st.caption(f"Colección: `{COLLECTION_NAME}`")
    st.caption(f"Índice: `{VECTOR_INDEX_NAME}`")
    st.caption("Embeddings: gemini-embedding-001 (3072 dimensiones)")
    st.caption(f"Modelo generativo: `{GENERATION_MODEL}`")
    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if total_docs == 0:
    st.warning("La colección está vacía. Ejecuta ingest.py o el notebook de Colab y vuelve a cargar la aplicación.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Ver fragmentos recuperados y scores"):
                for source in message["sources"]:
                    st.markdown(
                        f"**Fragmento {source.get('id', '?')}** · score: `{float(source.get('score', 0)):.4f}` · fuente: `{source.get('source', 'PDF')}`"
                    )
                    st.write(source.get("texto", ""))

question = st.chat_input("Escribe una pregunta sobre el PDF...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner(f"Buscando los {k} fragmentos más relevantes y generando la respuesta..."):
            try:
                answer, sources = answer_question(collection, client_genai, question, k=k, generation_model=GENERATION_MODEL)
                st.markdown(answer)
                if sources:
                    with st.expander("Ver fragmentos recuperados y scores", expanded=True):
                        for source in sources:
                            st.markdown(
                                f"**Fragmento {source.get('id', '?')}** · score: `{float(source.get('score', 0)):.4f}` · fuente: `{source.get('source', 'PDF')}`"
                            )
                            st.write(source.get("texto", ""))
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                error_message = (
                    "Ocurrió un error al consultar el índice vectorial o Gemini. "
                    "Comprueba que el índice esté en estado READY y que el nombre, la dimensión y los permisos sean correctos."
                )
                st.error(error_message)
                st.caption(f"Tipo de error: {type(exc).__name__}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )

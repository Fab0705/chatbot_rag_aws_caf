from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from google import genai
from google.genai import types

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 3072
BATCH_SIZE = 16


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    page_texts = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            page_texts.append(page_text)
        else:
            print(f"Aviso: la página {page_number} no devolvió texto; si contiene una imagen escaneada, requiere OCR.")
    text = "\n".join(page_texts).strip()
    if not text:
        raise ValueError("No se pudo extraer texto del PDF. Verifica si el documento está escaneado o protegido.")
    return text


def split_chunks(text: str, chunk_size: int = 1000) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que cero.")
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size) if text[i:i + chunk_size].strip()]


def embed_batch(client: genai.Client, batch: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=batch,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    vectors = [[float(v) for v in item.values] for item in response.embeddings]
    if len(vectors) != len(batch):
        raise RuntimeError("La cantidad de embeddings recibidos no coincide con la cantidad de fragmentos.")
    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Dimensión incorrecta: se esperaban {EMBEDDING_DIMENSIONS}, se recibieron {len(vector)}.")
    return vectors


def ensure_vector_index(collection, index_name: str) -> None:
    existing = list(collection.list_search_indexes())
    if any(index.get("name") == index_name for index in existing):
        print(f"El índice {index_name} ya existe. Verifica que su definición use 3072 dimensiones y similarity=dotProduct.")
        return
    model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": EMBEDDING_DIMENSIONS,
                    "similarity": "dotProduct",
                }
            ]
        },
        name=index_name,
        type="vectorSearch",
    )
    collection.create_search_index(model=model)
    print(f"Solicitud de creación enviada para el índice {index_name}.")
    print("La creación es asíncrona: espera a que Atlas muestre el estado READY antes de abrir el chatbot.")


def ingest_pdf(
    pdf_path: Path,
    mongodb_uri: str,
    database_name: str,
    collection_name: str,
    index_name: str,
    chunk_size: int = 1000,
) -> int:
    if not pdf_path.exists():
        raise FileNotFoundError(f"No se encontró el PDF: {pdf_path}")

    print(f"Leyendo: {pdf_path.name}")
    text = extract_pdf_text(pdf_path)
    chunks = split_chunks(text, chunk_size=chunk_size)
    sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    print(f"Texto extraído: {len(text):,} caracteres | Fragmentos: {len(chunks)} | Chunk size: {chunk_size}")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Define GOOGLE_API_KEY como variable de entorno antes de ejecutar la ingesta.")
    client_genai = genai.Client(api_key=api_key)

    mongo_client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=10000)
    try:
        mongo_client.admin.command("ping")

        database = mongo_client[database_name]

        if collection_name not in database.list_collection_names():
            database.create_collection(collection_name)

        collection = database[collection_name]

        ensure_vector_index(collection, index_name)

        # Prepara todos los vectores antes de sustituir la versión anterior del archivo.
        # Así, un fallo de Gemini no elimina una ingesta válida que ya exista.
        documents = []
        now = datetime.now(timezone.utc)
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]
            vectors = embed_batch(client_genai, batch)
            for offset, (chunk, vector) in enumerate(zip(batch, vectors)):
                chunk_id = start + offset
                documents.append(
                    {
                        "id": chunk_id,
                        "texto": chunk,
                        "embedding": vector,
                        "source": pdf_path.name,
                        "source_sha256": sha256,
                        "ingested_at": now,
                    }
                )
            print(f"Embeddings generados: {min(start + len(batch), len(chunks))}/{len(chunks)}")

        collection.delete_many({"source": pdf_path.name})
        if documents:
            collection.insert_many(documents, ordered=True)
        print(f"Ingesta finalizada: {len(documents)} fragmentos insertados en {database_name}.{collection_name}.")
        print(f"Índice vectorial: {index_name}. Confirma estado READY en MongoDB Atlas.")
        return len(documents)
    finally:
        mongo_client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingesta de un PDF a MongoDB Atlas Vector Search con Gemini.")
    parser.add_argument("pdf", type=Path, help="Ruta al PDF de origen.")
    parser.add_argument("--mongodb-uri", default=os.getenv("MONGODB_URI"), help="URI de MongoDB Atlas (o variable MONGODB_URI).")
    parser.add_argument("--database", default="pdf_embeddings_db")
    parser.add_argument("--collection", default="pdf_vectors")
    parser.add_argument("--index", default="vector_index")
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args()
    if not args.mongodb_uri:
        parser.error("Falta MONGODB_URI. Define la variable de entorno o pasa --mongodb-uri.")
    try:
        ingest_pdf(args.pdf, args.mongodb_uri, args.database, args.collection, args.index, args.chunk_size)
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

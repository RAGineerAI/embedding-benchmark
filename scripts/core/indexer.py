import time  # ← Импорт в начале файла
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_ollama import OllamaEmbeddings
from .config import QDRANT_URL, get_collection_name, OLLAMA_BASE_URL
from rank_bm25 import BM25Okapi


def create_collection(
    client: QdrantClient,
    collection_name: str,
    dimension: int
) -> None:
    """Создаёт коллекцию в Qdrant. Если существует — удаляет и создаёт заново."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
    )




def index_chunks(
        client: QdrantClient,
        model_name: str,
        chunks: list,
        dimension: int
) -> tuple[float, BM25Okapi]:
    """Индексирует чанки и возвращает (время, bm25_index)."""
    collection_name = get_collection_name(model_name)
    create_collection(client, collection_name, dimension)

    embeddings = OllamaEmbeddings(model=model_name, base_url=OLLAMA_BASE_URL)
    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)

    start_time = time.time()
    
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        point = PointStruct(
            id=i,
            vector=vector,
            payload={
                "page_content": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "model": model_name
            }
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)

    # 🆕 Создаём BM25-индекс
    tokenized_chunks = [chunk.page_content.split() for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)

    elapsed = time.time() - start_time
    return elapsed, bm25_index

if __name__ == "__main__":
    from .data import load_documents, chunk_documents
    
    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    
    documents = load_documents()
    chunks = chunk_documents(documents)
    
    elapsed, bm25_index = index_chunks(
        client=client,
        model_name="nomic-embed-text",
        chunks=chunks,
        dimension=768
    )
    
    print(f"✅ Индексация завершена за {elapsed:.2f} сек")
    print(f"🔍 BM25-индекс создан: {type(bm25_index).__name__}")
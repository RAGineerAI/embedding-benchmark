from qdrant_client import QdrantClient
import time
from langchain_ollama import OllamaEmbeddings
import sys
sys.path.insert(0, '.')
from scripts.core.config import get_collection_name, OLLAMA_BASE_URL
from qdrant_client.models import Filter, FieldCondition, MatchAny

"""
# 2. Создайте клиент
client = QdrantClient(url="http://localhost:6333")

# 3. Создайте эмбеддинги
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)

# 4. Векторизуйте запрос
query = "сумма договора"
query_vector = embeddings.embed_query(query)

print(type(query_vector))
print(len(query_vector))
print(query_vector[:5])  # Первые 5 чисел

result = client.query_points(
    collection_name=get_collection_name("nomic-embed-text"),
    query=query_vector,
    limit=3
)
points = result.points

print(type(points))
print(len(points))
print(points[0].score)  # Первый элемент
print(points[0].payload)  # Все поля объекта
point = result.points[0]
print(point.payload.keys())
"""

def dense_search(
    client: QdrantClient,
    model_name: str,
    query: str,
    k: int = 5,
    filter_sources: list = None  # ← список источников (опционально)
) -> tuple[list, float]:
    """Плотный поиск через Qdrant."""
    start_time = time.time()
    
    if not query:
        return [], 0.0
    
    collection_name = get_collection_name(model_name)
    embeddings = OllamaEmbeddings(model=model_name, base_url=OLLAMA_BASE_URL)
    query_vector = embeddings.embed_query(query)
    
    # Вариант 1: С фильтром (для hybrid search)
    if filter_sources:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchAny(any=filter_sources)
                )
            ]
        )
        result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=k,
            query_filter=query_filter
        )
    
    # Вариант 2: Без фильтра (обычный поиск по всей коллекции)
    else:
        result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=k
        )
    
    results = []
    for point in result.points:
        results.append({
            "content": point.payload.get("page_content", ""),
            "source": point.payload.get("source", "unknown"),
            "score": point.score
        })
    
    elapsed = time.time() - start_time
    return results, elapsed
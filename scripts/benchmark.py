"""
Embedding Benchmark — главный скрипт.

Сравнивает:
    - Dense Search (Qdrant)
    - BM25 Search (ключевые слова)
    - Hybrid Search (BM25 → Dense)

Для каждой модели и каждого метода считает метрики.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent))

from qdrant_client import QdrantClient
from tqdm import tqdm

from core.config import MODELS, QDRANT_URL, K_VALUES, METRICS_FILE
from core.data import load_queries, load_documents, chunk_documents
from core.indexer import index_chunks
from core.searcher import dense_search, bm25_search, hybrid_search
from core.metrics import calculate_all_metrics
from core.reporter import print_results_table, save_results


def run_benchmark_for_model(
    client: QdrantClient,
    model_name: str,
    model_config: dict,
    queries: list,
    chunks: list
) -> list:
    """
    Запускает бенчмарк для одной модели:
    индексирует, тестирует 3 метода поиска, считает метрики.
    
    Returns:
        Список результатов для каждого метода поиска.
    """
    print(f"\n{'='*60}")
    print(f"🔬 Модель: {model_name}")
    print(f"{'='*60}")

    # 1. Индексация
    elapsed_index, bm25_index = index_chunks(
        client,
        model_name,
        chunks,
        model_config["dimension"]
    )

    # 2. Тестируем три метода поиска
    methods = {
        "dense": lambda q: dense_search(client, model_name, q, k=5)[0],
        "bm25": lambda q: bm25_search(q, bm25_index, chunks, k=5),
        "hybrid": lambda q: hybrid_search(client, model_name, q, bm25_index, chunks, k=5),
    }

    results = []

    for method_name, search_fn in methods.items():
        print(f"  🔍 Метод: {method_name}")
        
        all_results = []
        for query in tqdm(queries, desc=f"    {method_name}"):
            found = search_fn(query["query"])
            all_results.append(found)

        metrics = calculate_all_metrics(all_results, queries, K_VALUES)

        results.append({
            "model": model_name,
            "method": method_name,
            "recall_at_1": metrics.get("recall_at_1", 0),
            "recall_at_3": metrics.get("recall_at_3", 0),
            "recall_at_5": metrics.get("recall_at_5", 0),
            "mrr": metrics.get("mrr", 0),
            "index_time_sec": round(elapsed_index, 2),
        })

    return results

def main():
    """Главная функция бенчмарка."""
    print("🚀 Запуск Embedding Benchmark")
    print("=" * 60)

    # 1. Загрузка данных
    print("\n📂 Загрузка данных...")
    queries = load_queries()
    documents = load_documents()
    chunks = chunk_documents(documents)

    # 2. Подключение к Qdrant
    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    print(f"🔌 Подключено к Qdrant: {QDRANT_URL}")

    # 3. Запуск для каждой модели
    all_results = []
    for model_name, model_config in MODELS.items():
        results = run_benchmark_for_model(
            client, model_name, model_config, queries, chunks
        )
        all_results.extend(results)

    # 4. Вывод результатов
    print_results_table(all_results)
    save_results(all_results, METRICS_FILE)

    print("\n✅ Готово!")


if __name__ == "__main__":
    main()
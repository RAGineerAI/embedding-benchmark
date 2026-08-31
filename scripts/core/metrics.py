"""
Метрики качества поиска.

Здесь считаем:
    - recall_at_k
    - mean_reciprocal_rank
    - calculate_all_metrics
"""

from typing import List, Dict, Any


def recall_at_k(results: List[Dict[str, Any]], relevant_doc: str, k: int) -> bool:
    """
    Проверяет, попал ли правильный документ в топ-K.
    
    Args:
        results: список результатов поиска
        relevant_doc: имя файла с правильным ответом
        k: сколько первых результатов проверять
    
    Returns:
        True, если документ найден в топ-K, иначе False
    """
    for result in results[:k]:
        if result["source"] == relevant_doc:
            return True
    return False


def mean_reciprocal_rank(results: List[Dict[str, Any]], relevant_doc: srt) -> float:
    """
    Вычисляет Reciprocal Rank для одного запроса.
    
    Формула: 1 / позиция правильного документа.
    Если не найден — возвращает 0.
    
    Args:
        results: список результатов поиска
        relevant_doc: имя файла с правильным ответом
    
    Returns:
        Значение от 0 до 1
    """
    for i, result in enumerate(results, start=1):
        if result["source"] == relevant_doc:
            return 1.0 / i
    return 0.0

def calculate_all_metrics(
        all_results: List[List[Dict[str, Any]]],
        queries: List[Dict[str, Any]],
        k_values: List[int]
) -> Dict[str, float]:
    """
    Считает метрики для всех запросов.
    
    Args:
        all_results: результаты поиска для каждого запроса
        queries: эталонные запросы (с relevant_doc)
        k_values: значения k (например, [1, 3, 5])
    
    Returns:
        Словарь с метриками
    """
    n = len(queries)
    metrics = {}

    for k in k_values:
        correct = 0
        for results, query in zip(all_results, queries):
            if recall_at_k(results, query["relevant_doc"], k):
                correct+=1
        metrics[f"recall_at_{k}"] = round(correct / n, 4)

    total_mrr = 0.0
    for results, query in zip(all_results, queries):
        total_mrr += mean_reciprocal_rank(results, query["relevant_doc"])
    metrics["mrr"] = round(total_mrr / n, 4)

    return metrics

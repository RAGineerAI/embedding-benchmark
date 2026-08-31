"""
Вывод результатов бенчмарка.

Здесь:
    - print_results_table
    - save_results
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def print_results_table(results: List[Dict[str, Any]]) -> None:
    """
    Печатает таблицу с результатами в консоль.
    """
    print(f"\n{'='*70}")
    print("РЕЗУЛЬТАТЫ БЕНЧМАРКА")
    print(f"{'='*70}")

    header = (
        f"{'Модель':<25} "
        f"{'Recall@1':>10} "
        f"{'Recall@5':>10} "
        f"{'MRR':>10} "
        f"{'Время(сек)':>12}"
    )

    print(header)
    print("-" * 70)

    for r in results:
        row = (
            f"{r['model']:<25} "
            f"{r.get('recall_at_1', 0):>10.4f} "
            f"{r.get('recall_at_5', 0):>10.4f} "
            f"{r.get('mrr', 0):>10.4f} "
            f"{r.get('index_time_sec', 0):>12.2f}"
        )
        print(row)

    print(f"{'='*70}\n")

def save_results(results: List[Dict[str, Any]], file_path: Path) -> None:
    """
    Сохраняет результаты в JSON-файл.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"💾 Результаты сохранены: {file_path}")
# 🔬 Embedding Benchmark

[![Status](https://img.shields.io/badge/status-MVP_completed-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)]()
[![Qdrant](https://img.shields.io/badge/Qdrant-✓-orange.svg)]()
[![Ollama](https://img.shields.io/badge/Ollama-✓-green.svg)]()

Сравнение embedding-моделей и методов поиска для RAG-систем на финансовых документах.

**Проект 2.5 из дорожной карты перехода в AI Engineer.**
## 🎯 Что решает

Бенчмарк отвечает на вопрос: **какой метод поиска лучше для RAG на финансовых документах?**

Сравнивает:
- **Dense Search** — поиск по смыслу (векторы Qdrant)
- **BM25 Search** — поиск по ключевым словам (цифры, коды)
- **Hybrid Search** — каскад: BM25 → Dense

## 🛠️ Стек

| Компонент        | Технология                |
| ---------------- | ------------------------- |
| **Язык**         | Python 3.12               |
| **Векторная БД** | Qdrant                    |
| **LLM-движок**   | Ollama (nomic-embed-text) |
| **BM25**         | rank_bm25                 |
| **Метрики**      | Recall@k, MRR             |
| **Визуализация** | pandas                    |
## 🏗️ Архитектура
Документы (PDF, DOCX, TXT)  
↓  
[Чанкинг] → 500 символов, overlap 50  
↓  
[Два индекса]  
├── Qdrant (векторы, Dense)  
└── BM25 (слова, Sparse)  
↓  
[Поиск: 3 метода]  
├── Dense → по смыслу  
├── BM25 → по словам  
└── Hybrid → каскад  
↓  
[Метрики] → Recall@k, MRR  
↓  
[Отчёт] → таблица + JSON
## 🚀 Быстрый старт

### Требования

- Python 3.12+
- Docker
- Ollama
- Qdrant
### Установка

```bash
# 1. Клонировать репозиторий
git clone https://github.com/RAGineerAI/embedding-benchmark.git
cd embedding-benchmark

# 2. Создать окружение
python3 -m venv venv
source venv/bin/activate

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Запустить Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 5. Запустить Ollama и загрузить модель
ollama pull nomic-embed-text

# 6. Запустить бенчмарк
python scripts/benchmark.py
```

---

## 📝 Пояснение

### Архитектура

| Часть           | Зачем                                                           |
| --------------- | --------------------------------------------------------------- |
| **Путь данных** | Видно, как данные текут (документы → чанки → индексы → метрики) |
| **Два индекса** | Показывает понимание разницы между Dense и Sparse               |
| **3 метода**    | Гордость проекта — сравнение                                    |

### Быстрый старт

| Блок           | Что делает                                                     |
| -------------- | -------------------------------------------------------------- |
| **Требования** | Что установить заранее                                         |
| **Установка**  | Пошагово: клон → venv → зависимости → Qdrant → Ollama → запуск |
|                |                                                                |
## 📁 Структура проекта

embedding-benchmark/  
├── scripts/  
│ ├── core/  
│ │ ├── config.py # Настройки  
│ │ ├── data.py # Загрузка данных  
│ │ ├── indexer.py # Индексация + BM25  
│ │ ├── searcher.py # 3 метода поиска  
│ │ ├── metrics.py # Recall@k, MRR  
│ │ └── reporter.py # Таблица + JSON  
│ └── benchmark.py # Оркестратор  
├── dataset/  
│ ├── documents/ # Тестовые документы  
│ └── queries.json # Запросы + эталоны  
├── results/  
│ └── metrics.json # Результаты  
└── requirements.txt
## 📊 Результаты

На датасете из **5 финансовых документов** и **10 запросов**:

| Метод | Recall@1 | Recall@5 | MRR | Время (сек) |
|-------|----------|----------|-----|-------------|
| **Dense** | 0.70 | 1.00 | 0.80 | 0.01 |
| **BM25** | 0.60 | 1.00 | 0.71 | 0.01 |
| **Hybrid** | 0.70 | 1.00 | 0.80 | 0.01 |

## 💡 Выводы

1. **Dense и Hybrid** показали одинаковый результат на маленьком датасете
2. **BM25** отстаёт по Recall@1 и MRR
3. **Recall@5 = 1.00 у всех** — все правильные документы найдены в топ-5
4. Для оценки преимущества Hybrid нужен датасет **100+ документов**

## 👤 Автор

**RAGineerAI** — [GitHub](https://github.com/RAGineerAI)

## 📄 Лицензия

MIT
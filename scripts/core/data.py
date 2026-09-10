"""
Загрузка тестовых данных: запросов и документов.

Отдельный модуль, чтобы не смешивать логику загрузки с логикой бенчмарка.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_SIZE, CHUNK_OVERLAP, QUERIES_FILE, DOCUMENT_DIR


def load_queries(queries_file: Path = None) -> List[Dict[str, Any]]:
    """
    Загружает тестовые запросы из JSON-файла.
    
    Возвращает список словарей с ключами:
        - id: идентификатор запроса
        - query: текст запроса
        - relevant_doc: имя файла с правильным ответом
        - expected_answer: эталонный ответ
    """
    file_path = queries_file or QUERIES_FILE

    with open(file_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"📋 Загружено запросов: {len(queries)}")
    return queries


def load_documents(docs_folder: Path = None) -> List[Any]:
    """
    Загружает все текстовые документы из папки.
    
    Возвращает список LangChain Document objects.
    """
    docs_path = docs_folder or DOCUMENT_DIR
    documents = []
    
    for file_path in docs_path.glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        
        # Добавляем имя файла в метаданные
        for doc in docs:
            doc.metadata["source"] = file_path.name

        documents.extend(docs)

    print(f"📄 Загружено документов: {len(documents)}")
    return documents

def chunk_documents(documents: List[Any]) -> List[Any]:
    """
    Разбивает документы на чанки.
    
    Возвращает список чанков (также LangChain Document objects).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"✂️ Создано чанков: {len(chunks)}")
    
    return chunks
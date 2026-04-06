import pytest
from src.ingestion.arxiv_crawler import format_arxiv_query

def test_format_arxiv_query():
    # Kiểm tra xem hàm format query có tạo ra chuỗi đúng cấu trúc không
    query = "machine learning"
    max_results = 10
    formatted = format_arxiv_query(query, max_results)
    
    assert "search_query=all:machine+learning" in formatted
    assert "max_results=10" in formatted
    assert "export.arxiv.org/api/query" in formatted

def test_placeholder():
    assert 1 + 1 == 2

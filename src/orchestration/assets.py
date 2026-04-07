import os
import logging
from dagster import asset, AssetExecutionContext
from src.ingestion.arxiv_crawler import get_minio_client as crawler_minio, setup_minio_buckets as setup_raw, crawl_arxiv_papers
from src.processing.pdf_parser import get_minio_client as parser_minio, setup_minio_buckets as setup_parsed, parse_pdfs_to_markdown
from src.processing.chunker import get_minio_client as chunker_minio, setup_minio_buckets as setup_chunker, chunk_markdown_files
from src.processing.cleaner import get_minio_client as cleaner_minio, setup_minio_buckets as setup_cleaner, clean_chunks_layer
from src.processing.classifier import get_minio_client as classifier_minio, setup_minio_buckets as setup_classifier, classify_chunks_layer
from src.database.dataset_generator import generate_finetune_data

# Cấu hình logging
logging.basicConfig(level=logging.INFO)

@asset(group_name="arxiv_pipeline")
def arxiv_raw_data(context: AssetExecutionContext):
    """Bước 1: Cào dữ liệu PDF và Metadata từ ArXiv với đầy đủ các chủ đề."""
    import time
    client = crawler_minio()
    setup_raw(client)
    
    topic_queries = [
        "cat:cs.AI", "cat:cs.LG",           # 1. AI/ML
        "cat:cs.CR",                        # 2. Cryptography
        "cat:cs.IT", "cat:cs.NI",           # 3. Network & Info Theory
        "cat:eess.SY", "cat:eess.SP",       # 4. Signal Processing
        "cat:q-fin.GN", "cat:q-fin.CP",     # 5. Quant Finance
        "all:military OR all:defense"       # 6. Defense
    ]
    
    for query in topic_queries:
        try:
            context.log.info(f"Crawl Topic: {query}")
            crawl_arxiv_papers(client, query=query, max_results=5)
            # Increase standard pause between topics to 10 seconds
            context.log.info("Rate limit safeguard engaged. Pausing execution for 10 seconds...")
            time.sleep(10)
        except Exception as e:
            # If a topic fails (usually due to 429/503), log a warning and wait longer
            context.log.warning(f"ArXiv request failed for topic {query}: {e}")
            context.log.info("Potential rate limit or server issue. Cooling down for 60 seconds...")
            time.sleep(60)
    
    context.log.info("ArXiv Raw Data Asset Materialized for all topics.")

@asset(deps=[arxiv_raw_data], group_name="arxiv_pipeline")
def parsed_markdown(context: AssetExecutionContext):
    """Bước 2: Chuyển đổi PDF sang Markdown sử dụng Docling."""
    client = parser_minio()
    setup_parsed(client)
    parse_pdfs_to_markdown(client=client)
    context.log.info("Parsed Markdown Asset Materialized.")

@asset(deps=[parsed_markdown], group_name="arxiv_pipeline")
def semantic_chunks(context: AssetExecutionContext):
    """Bước 3: Chia nhỏ văn bản dựa trên ngữ nghĩa (Semantic Chunking)."""
    client = chunker_minio()
    setup_chunker(client)
    chunk_markdown_files(client=client)
    context.log.info("Semantic Chunks Asset Materialized.")

@asset(deps=[semantic_chunks], group_name="arxiv_pipeline")
def cleaned_chunks(context: AssetExecutionContext):
    """Bước 4: Làm sạch và chuẩn hóa văn bản (Cleaner)."""
    client = cleaner_minio()
    setup_cleaner(client)
    clean_chunks_layer(client=client)
    context.log.info("Cleaned Chunks Asset Materialized.")

@asset(deps=[cleaned_chunks], group_name="arxiv_pipeline")
def classified_chunks(context: AssetExecutionContext):
    """Bước 5: Phân loại chủ đề AI (Classifier)."""
    client = classifier_minio()
    setup_classifier(client)
    classify_chunks_layer(client=client)
    context.log.info("Classified Chunks Asset Materialized.")

@asset(deps=[classified_chunks], group_name="arxiv_pipeline")
def finetune_dataset(context: AssetExecutionContext):
    """Bước 6: Sinh Dataset hoàn chỉnh Q&A (Generator)."""
    import asyncio
    asyncio.run(generate_finetune_data())
    context.log.info("Finetune Dataset Asset Materialized.")
@asset(deps=[finetune_dataset], group_name="arxiv_pipeline")
def latest_dataset_showcase(context: AssetExecutionContext):
    """Bước 7: Xuất dữ liệu mới nhất ra thư mục showcase để commit lên GitHub."""
    client = crawler_minio()
    bucket = "finetune-data"
    obj_name = "dataset/finetune_dataset.jsonl"
    # Đường dẫn trong container (đã được mount ra showcase/ trên host)
    dest_path = "/app/showcase/latest_dataset.jsonl"
    
    try:
        client.fget_object(bucket, obj_name, dest_path)
        context.log.info(f"Successfully exported {obj_name} to {dest_path}")
    except Exception as e:
        context.log.error(f"Failed to export showcase dataset: {e}")
        raise e

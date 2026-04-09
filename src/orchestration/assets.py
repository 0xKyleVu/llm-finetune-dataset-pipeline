import os
import logging
from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue, asset_check, AssetCheckResult, AssetCheckSeverity
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
            crawl_arxiv_papers(client, query=query, max_results=3)
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
    stats = classify_chunks_layer(client=client)

    if not stats:
        context.log.warning("Classification produced no stats. Skipping metadata render.")
        return

    # --- Tạo bảng Markdown Category Distribution ---
    total = stats["total_chunks"]
    cat_table_rows = "\n".join([
        f"| {label} | {count} | {count/total*100:.1f}% | {stats['confidence_by_label'].get(label, 0):.3f} |"
        for label, count in sorted(stats["category_counts"].items(), key=lambda x: -x[1])
    ])
    category_table_md = (
        "| Category | Chunks | % of Total | Avg Confidence |\n"
        "|---|---|---|---|\n"
        + cat_table_rows
    )

    # --- Tạo bảng Markdown Quality Tier ---
    tier_rows = "\n".join([
        f"| {tier} | {stats['tier_counts'].get(tier, 0)} | {stats['tier_counts'].get(tier, 0)/total*100:.1f}% |"
        for tier in ["High", "Medium", "Low"]
    ])
    tier_table_md = (
        "| Quality Tier | Chunks | % of Total |\n"
        "|---|---|---|\n"
        + tier_rows
    )

    context.log.info(f"Classified Chunks Asset Materialized. Status: {stats['status']}")

    return MaterializeResult(
        metadata={
            "Total Chunks": MetadataValue.int(total),
            "Usable Data (High + Medium)": MetadataValue.text(f"{stats['usable_pct']}%"),
            "Overall Confidence Score": MetadataValue.float(stats["overall_conf"]),
            "Model Status": MetadataValue.text(stats["status"]),
            "Category Distribution": MetadataValue.md(category_table_md),
            "Quality Tier Breakdown": MetadataValue.md(tier_table_md),
        }
    )


@asset(deps=[classified_chunks], group_name="arxiv_pipeline")
def finetune_dataset(context: AssetExecutionContext):
    """Bước 6: Sinh Dataset hoàn chỉnh Q&A (Generator)."""
    import asyncio
    asyncio.run(generate_finetune_data())
    context.log.info("Finetune Dataset Asset Materialized.")
@asset(deps=[finetune_dataset], group_name="arxiv_pipeline")
def latest_dataset_showcase(context: AssetExecutionContext):
    """Bước 7: Xuất dữ liệu mới nhất ra thư mục showcase để commit lên GitHub."""
    import subprocess
    client = crawler_minio()
    bucket = "finetune-data"
    obj_name = "dataset/finetune_dataset.jsonl"
    
    # Xác định đường dẫn gốc
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    dest_path = os.path.join(project_root, "showcase", "latest_dataset.jsonl")
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    try:
        # Tải file từ MinIO
        client.fget_object(bucket, obj_name, dest_path)
        context.log.info(f"Successfully exported {obj_name} to {dest_path}")
        
        # Git Add
        try:
            subprocess.run(["git", "add", dest_path], check=True, cwd=project_root)
            context.log.info(f"Auto-staged {dest_path} for Git commit.")
        except Exception as git_err:
            context.log.warning(f"Git add failed (check if git is installed and path is correct): {git_err}")
            
    except Exception as e:
        context.log.error(f"Failed to export showcase dataset: {e}")
        raise e

# ==============================================================================
# ASSET CHECKS (DATA QUALITY GATES)
# ==============================================================================

@asset_check(asset=arxiv_raw_data, description="Verify that raw PDF and metadata were successfully ingested.")
def check_raw_data_presence():
    client = crawler_minio()
    try:
        pdfs = list(client.list_objects("raw-data", prefix="pdf/"))
        metas = list(client.list_objects("raw-data", prefix="metadata/"))
        
        has_data = len(pdfs) > 0 and len(metas) > 0
        return AssetCheckResult(
            passed=has_data,
            metadata={"Total PDFs": len(pdfs), "Total Metadata files": len(metas)},
            severity=AssetCheckSeverity.WARN
        )
    except Exception as e:
        return AssetCheckResult(passed=False, description=str(e), severity=AssetCheckSeverity.ERROR)

@asset_check(asset=parsed_markdown, description="Verify the integrity of parsed markdown files.")
def check_markdown_integrity():
    client = parser_minio()
    try:
        mds = list(client.list_objects("processed-data", prefix="markdown/"))
        if not mds:
             return AssetCheckResult(passed=False, description="No markdown files found.", severity=AssetCheckSeverity.WARN)
             
        # Check first file as sample
        response = client.get_object("processed-data", mds[0].object_name)
        content = response.read().decode('utf-8')
        response.close()
        response.release_conn()
        
        is_healthy = len(content) > 100
        return AssetCheckResult(
            passed=is_healthy,
            description="Markdown files seem valid." if is_healthy else "Sample markdown file is too short/empty.",
            metadata={"Sample File Size": len(content)},
            severity=AssetCheckSeverity.WARN
        )
    except Exception as e:
        return AssetCheckResult(passed=False, description=str(e), severity=AssetCheckSeverity.ERROR)

@asset_check(asset=classified_chunks, description="Validate the health of the classified dataset.")
def check_dataset_health():
    client = classifier_minio()
    from src.processing.classifier import generate_classification_report
    stats = generate_classification_report(client)
    if not stats:
        return AssetCheckResult(passed=False, description="No stats generated.", severity=AssetCheckSeverity.WARN)
        
    usable = stats["usable_pct"]
    is_healthy = usable >= 50.0
    return AssetCheckResult(
        passed=is_healthy,
        description="Dataset health is within acceptable bounds." if is_healthy else f"Usable data dropped below 50% (Currently {usable}%).",
        metadata={"Usable Data %": usable},
        severity=AssetCheckSeverity.WARN
    )

@asset_check(asset=finetune_dataset, description="Ensure final finetune dataset is valid JSONL with no duplicates.")
def check_final_dataset_format():
    import json
    client = crawler_minio()
    try:
        response = client.get_object("finetune-data", "dataset/finetune_dataset.jsonl")
        content = response.read().decode('utf-8')
        response.close()
        response.release_conn()
        
        lines = [line for line in content.splitlines() if line.strip()]
        if not lines:
            return AssetCheckResult(passed=False, description="Final dataset is empty.", severity=AssetCheckSeverity.WARN)
            
        instructions = set()
        duplicates = 0
        for line in lines:
            try:
                data = json.loads(line)
                inst = data.get("finetune_sample", {}).get("instruction", "")
                if inst in instructions:
                    duplicates += 1
                instructions.add(inst)
            except Exception as json_err:
                return AssetCheckResult(passed=False, description=f"Invalid JSON format found: {json_err}", severity=AssetCheckSeverity.ERROR)
            
        is_valid = len(lines) >= 10 and duplicates == 0
        return AssetCheckResult(
            passed=is_valid,
            description="Valid JSONL dataset." if is_valid else f"Dataset validation failed: {duplicates} duplicates, {len(lines)} lines.",
            metadata={"Total Samples": len(lines), "Duplicates": duplicates},
            severity=AssetCheckSeverity.WARN
        )
    except Exception as e:
        return AssetCheckResult(passed=False, description=str(e), severity=AssetCheckSeverity.ERROR)

import os
import json
import logging
import io
from typing import List, Dict
from minio import Minio
from minio.error import S3Error

# Langchain Document Splitters
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Cấu hình Global Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CLOUD CONFIGURATION ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

BUCKET_NAME = "processed-data"

def get_minio_client() -> Minio:
    """Khởi tạo kết nối đến Data Lake MinIO."""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )

def setup_minio_buckets(client: Minio):
    """Đảm bảo Bucket đích tồn tại trên MinIO."""
    try:
        if not client.bucket_exists(BUCKET_NAME):
            client.make_bucket(BUCKET_NAME)
            logging.info(f"[Initialization] Created data bucket: s3://{BUCKET_NAME}")
        else:
            logging.info(f"[Initialization] Data bucket ready: s3://{BUCKET_NAME}")
    except S3Error as e:
        logging.error(f"[Initialization] MinIO Server error: {e}")
        raise

def is_object_exists(client: Minio, bucket_name: str, object_name: str) -> bool:
    """Kiểm tra sự tồn tại của object trên MinIO để đảm bảo tính Idempotency."""
    try:
        client.stat_object(bucket_name, object_name)
        return True
    except S3Error as e:
        if e.code == "NoSuchKey":
            return False
        raise

def chunk_markdown_files(client: Minio):
    """
    Sử dụng Langchain để chia mảng semantic chunking các file Markdown lấy trực tiếp từ MinIO.
    Lọc bỏ bớt các thông tin rác học thuật và đẩy JSON arrays trở về Data Lake.
    """
    # Chunk bài báo dựa trên semantic Markdown headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # Chunk text theo ký tự đệ quy (Recursive split 1000 char threshold)
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    try:
        md_objects = list(client.list_objects(BUCKET_NAME, prefix="markdown/"))
    except S3Error as e:
        logging.error(f"[Chunker] Failed to list objects in '{BUCKET_NAME}': {e}")
        return

    if not md_objects:
        logging.warning(f"[Chunker] No Markdown files found in s3://{BUCKET_NAME}/markdown/")
        return

    logging.info(f"[Chunker] Detected {len(md_objects)} Markdown documents. Initializing Langchain Processors...")

    for obj in md_objects:
        object_name = obj.object_name
        file_name = os.path.basename(object_name)
        base_name = os.path.splitext(file_name)[0]
        output_json_object = f"chunks/{base_name}_chunks.json"
        
        if is_object_exists(client, BUCKET_NAME, output_json_object):
            logging.info(f"  -> [Idempotency] Chunks exist, skipping: {output_json_object}")
            continue

        logging.info(f"[Chunker] Segmenting document layout: {file_name}")
        
        try:
            # Rút text Markdown đẩy vào RAM
            response = client.get_object(BUCKET_NAME, object_name)
            markdown_text = response.read().decode('utf-8')
            response.close()
            response.release_conn()

            # Bước 1: Chunk theo header
            md_header_splits = markdown_splitter.split_text(markdown_text)

            # Bước 2: Def rác (Ví dụ khoản References, Acknowledgments)
            NOISE_HEADERS = {
                "references", "bibliography", "acknowledgments",
                "acknowledgements", "appendix", "author contributions",
                "competing interests", "funding", "disclosure"
            }
            before_count = len(md_header_splits)
            md_header_splits = [
                doc for doc in md_header_splits
                if not any(
                    doc.metadata.get(h, "").strip().lower() in NOISE_HEADERS
                    for h in ("Header 1", "Header 2", "Header 3")
                )
            ]
            filtered_count = before_count - len(md_header_splits)
            if filtered_count > 0:
                logging.info(f"  -> [Filter] Erased {filtered_count} noise segments (Bibliography/Legal context).")

            # Bước 3: Chunk text theo ký tự đệ quy (Recursive split 1000 char threshold)
            final_chunks = char_splitter.split_documents(md_header_splits)
            
            # Bước 4: Chuyển metadata attribute context qua file Dict cho Python JSON
            processed_chunks = []
            for i, chunk in enumerate(final_chunks):
                processed_chunks.append({
                    "chunk_id": f"{base_name}_{i}",
                    "paper_id": base_name,
                    "metadata": chunk.metadata,
                    "content": chunk.page_content,
                    "char_length": len(chunk.page_content)
                })

            # Stream payload luồng ghi bộ lưu đệm RAM tới đích Cloud
            json_bytes = json.dumps(processed_chunks, ensure_ascii=False, indent=2).encode('utf-8')
            json_stream = io.BytesIO(json_bytes)
            
            client.put_object(
                BUCKET_NAME,
                output_json_object,
                data=json_stream,
                length=len(json_bytes),
                content_type="application/json"
            )

            logging.info(f"  -> [Chunker] Generated {len(processed_chunks)} vector-ready blocks. Uploaded to s3://{BUCKET_NAME}/{output_json_object}")

        except Exception as e:
            logging.error(f"[Chunker] Critical error segmenting file '{file_name}': {e}")

if __name__ == "__main__":
    logging.info("=========================================")
    logging.info("[System] STARTING CHUNKING LAYER")
    logging.info("=========================================")
    
    minio_client = get_minio_client()
    setup_minio_buckets(minio_client)
    
    chunk_markdown_files(client=minio_client)
    
    logging.info("[System] Chunking pipeline execution completed.")

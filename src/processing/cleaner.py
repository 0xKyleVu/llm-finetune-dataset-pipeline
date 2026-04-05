import os
import json
import logging
import re
import hashlib
import io
import ftfy
from minio import Minio
from minio.error import S3Error

# Cấu hình Global Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CLOUD CONFIGURATION ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

SOURCE_BUCKET = "processed-data"
DEST_BUCKET = "cleaned-data"

def get_minio_client() -> Minio:
    """Khởi tạo kết nối đến MinIO Data Lake."""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )

def setup_minio_buckets(client: Minio):
    """Đảm bảo Bucket đích tồn tại trên MinIO."""
    try:
        if not client.bucket_exists(DEST_BUCKET):
            client.make_bucket(DEST_BUCKET)
            logging.info(f"[Initialization] Created destination bucket: s3://{DEST_BUCKET}")
        else:
            logging.info(f"[Initialization] Destination bucket ready: s3://{DEST_BUCKET}")
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

def clean_text(text: str) -> str:
    """
    Thực hiện tiền xử lý NLP và sanitize văn bản tối ưu hóa cho huấn luyện LLM.
    """
    if not text:
        return ""
        
    # Bước 1: Fix Unicode
    text = ftfy.fix_text(text)
    
    # Bước 2: Loại bỏ các citation tham chiếu (e.g., [12], [3, 4], [1-5])
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    text = re.sub(r'\[\d+(?:-\d+)*\]', '', text)
    
    # Bước 3: Loại bỏ URL web (Giữ lại chuỗi mã DOI cho metadata)
    text = re.sub(r'https?://(?!doi\.org)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Bước 4: Loại bỏ whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()

def clean_chunks_layer(client: Minio):
    """
    Tải chuỗi Json chunk từ Data Lake về Memory, xử lý văn bản chuyên sâu
    áp dụng mã băm loại trùng lặp (global deduplication) và bắn đẩy Data đầu ra sang bucket chuẩn.
    """
    try:
        chunk_objects = list(client.list_objects(SOURCE_BUCKET, prefix="chunks/"))
    except S3Error as e:
        logging.error(f"[Cleaner] Failed to list objects in '{SOURCE_BUCKET}': {e}")
        return

    if not chunk_objects:
        logging.warning(f"[Cleaner] No JSON array files found in s3://{SOURCE_BUCKET}/chunks/")
        return

    logging.info(f"[Cleaner] Detected {len(chunk_objects)} document chunks. Initializing Sanitizer engine...")

    for obj in chunk_objects:
        object_name = obj.object_name
        file_name = os.path.basename(object_name)
        output_object_name = f"chunks/{file_name}"
        
        if is_object_exists(client, DEST_BUCKET, output_object_name):
            logging.info(f"  -> [Idempotency] Cleaned array exists, skipping: {output_object_name}")
            continue

        logging.info(f"[Cleaner] Sanitizing arrays for document: {file_name}")
        
        try:
            # Rút bản Json thô từ Stream Minio Array
            response = client.get_object(SOURCE_BUCKET, object_name)
            chunks_data = json.loads(response.read().decode('utf-8'))
            response.close()
            response.release_conn()
            
            cleaned_chunks = []
            seen_hashes = set()
            dup_count = 0
            
            for chunk in chunks_data:
                original_content = chunk.get("content", "")
                sanitized_content = clean_text(original_content)
                
                # Quét lại sau khi lọc block Empty text zero 
                if not sanitized_content:
                    continue
                
                # Global Deduplication Filter (Chống trùng lặp)
                content_hash = hashlib.md5(sanitized_content.encode()).hexdigest()
                if content_hash in seen_hashes:
                    dup_count += 1
                    continue
                seen_hashes.add(content_hash)
                    
                chunk["content"] = sanitized_content
                chunk["char_length"] = len(sanitized_content)
                cleaned_chunks.append(chunk)
            
            if dup_count > 0:
                logging.info(f"  -> [Filter] Discarded {dup_count} duplicated chunks via cryptographic hash.")

            # Upload batch mới vào MinIO bucket
            json_bytes = json.dumps(cleaned_chunks, ensure_ascii=False, indent=2).encode('utf-8')
            json_stream = io.BytesIO(json_bytes)
            
            client.put_object(
                DEST_BUCKET,
                output_object_name,
                data=json_stream,
                length=len(json_bytes),
                content_type="application/json"
            )

            logging.info(f"  -> [Cleaner] Sanitized {len(cleaned_chunks)} authentic blocks. Linked to s3://{DEST_BUCKET}/{output_object_name}")

        except Exception as e:
            logging.error(f"[Cleaner] Critical failure auditing document '{file_name}': {e}")

if __name__ == "__main__":
    logging.info("=========================================")
    logging.info("[System] STARTING DATA CLEANING LAYER")
    logging.info("=========================================")
    
    minio_client = get_minio_client()
    setup_minio_buckets(minio_client)
    
    clean_chunks_layer(client=minio_client)
    
    logging.info("[System] Cleaning pipeline execution completed.")

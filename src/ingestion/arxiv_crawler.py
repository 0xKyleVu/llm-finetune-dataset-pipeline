import arxiv
import os
import json
import logging
import ssl
import time
import io
import tempfile
from typing import List, Dict
from minio import Minio
from minio.error import S3Error

# Cấu hình SSL: Ưu tiên dùng 'certifi', fallback về unverified context nếu không tìm thấy.
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
except ImportError:
    ssl._create_default_https_context = ssl._create_unverified_context
    logging.warning("[System] Certificate package 'certifi' not found. Disabling SSL verification. Recommended: pip install certifi")

# Cấu hình Global Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CLOUD CONFIGURATION ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
BUCKET_NAME = "raw-data"

def get_minio_client() -> Minio:
    """Khởi tạo kết nối đến hệ thống Data Lake MinIO."""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )

def setup_minio_buckets(client: Minio):
    """Đảm bảo Bucket lưu trữ đã tồn tại trên MinIO."""
    try:
        if not client.bucket_exists(BUCKET_NAME):
            client.make_bucket(BUCKET_NAME)
            logging.info(f"[Initialization] Created Data Lake bucket: s3://{BUCKET_NAME}")
        else:
            logging.info(f"[Initialization] Data Lake bucket already exists: s3://{BUCKET_NAME}")
    except S3Error as e:
        logging.error(f"[Initialization] MinIO server error: {e}")
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

def crawl_arxiv_papers(client: Minio, query: str, max_results: int = 10):
    """
    Tìm kiếm thông minh từ ArXiv qua API và stream trực tiếp lên MinIO Data Lake.
    
    Args:
        client (Minio): Object kết nối MinIO.
        query (str): Lệnh tìm kiếm nâng cao trên ArXiv.
        max_results (int): Giới hạn số lượng paper cần lấy.
    """
    logging.info(f"[Ingestion] Commencing search query: '{query}', max results: {max_results}")
    
    # Increase delay and retries to be more conservative and avoid HTTP 429/503
    arxiv_client = arxiv.Client(page_size=max_results, delay_seconds=10, num_retries=10)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers_processed = 0
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for entry in arxiv_client.results(search):
            paper_id = entry.get_short_id()
            logging.info(f"[Ingestion] Processing academic paper: [{paper_id}] {entry.title}")
            
            # --- Bước 1: Tổng hợp và Upload Metadata từ memory ---
            metadata = {
                "id": paper_id,
                "title": entry.title,
                "authors": [author.name for author in entry.authors],
                "summary": entry.summary,
                "published": entry.published.isoformat(),
                "primary_category": entry.primary_category,
                "categories": entry.categories,
                "pdf_url": entry.pdf_url
            }
            
            meta_object_name = f"metadata/{paper_id}.json"
            if not is_object_exists(client, BUCKET_NAME, meta_object_name):
                # Mã hóa Dict thành luồng bytes JSON
                meta_bytes = json.dumps(metadata, ensure_ascii=False, indent=2).encode('utf-8')
                meta_stream = io.BytesIO(meta_bytes)
                client.put_object(
                    BUCKET_NAME, 
                    meta_object_name, 
                    data=meta_stream, 
                    length=len(meta_bytes),
                    content_type="application/json"
                )
                logging.info(f"  -> Successfully uploaded metadata to: s3://{BUCKET_NAME}/{meta_object_name}")
            else:
                logging.info(f"  -> [Idempotency] Metadata already exists, skipping: s3://{BUCKET_NAME}/{meta_object_name}")

            # --- Bước 2: Tải PDF qua buffer trung gian và đẩy lên MinIO ---
            pdf_object_name = f"pdf/{paper_id}.pdf"
            if not is_object_exists(client, BUCKET_NAME, pdf_object_name):
                logging.info(f"  Downloading PDF stream from ArXiv servers...")
                entry.download_pdf(dirpath=temp_dir, filename=f"{paper_id}.pdf")
                tmp_pdf_path = os.path.join(temp_dir, f"{paper_id}.pdf")
                
                logging.info(f"  Uploading PDF payload to MinIO Data Lake...")
                client.fput_object(
                    BUCKET_NAME, 
                    pdf_object_name, 
                    tmp_pdf_path,
                    content_type="application/pdf"
                )
                logging.info(f"  -> Successfully uploaded PDF to: s3://{BUCKET_NAME}/{pdf_object_name}")
                
                # Cleanup file tmp
                os.remove(tmp_pdf_path)
            else:
                logging.info(f"  -> [Idempotency] PDF already exists, skipping: s3://{BUCKET_NAME}/{pdf_object_name}")
            
            papers_processed += 1
            
    logging.info(f"[Ingestion] Completed Ingestion array for query '{query}'. Total documents: {papers_processed}.")

if __name__ == "__main__":
    logging.info("=========================================")
    logging.info("[System] STARTING INGESTION LAYER")
    logging.info("=========================================")
    
    minio_client = get_minio_client()
    setup_minio_buckets(minio_client)
    
    topic_queries = [
        "cat:cs.AI", "cat:cs.LG",           # 1. Trí tuệ nhân tạo (AI/ML)
        "cat:cs.CR",                        # 2. Bảo mật & Mã hóa (Cryptography)
        "cat:cs.IT", "cat:cs.NI",           # 3. Mạng Hệ thống & Information Theory (Network & Information Theory)
        "cat:eess.SY", "cat:eess.SP",       # 4. Kỹ thuật hệ thống & Xử lý tín hiệu (Signal Processing)
        "cat:q-fin.GN", "cat:q-fin.CP",     # 5. Tài chính định lượng (Quant Finance)
        "all:military OR all:defense"       # 6. Khu vực Quốc phòng Không gian sự kiện (Military)
    ]
    
    for query in topic_queries:
        logging.info(f"\n[Ingestion] CRAwLING TOPIC: {query}")
        crawl_arxiv_papers(
            client=minio_client,
            query=query, 
            max_results=2  # Chỉ chạy lấy số lượng nhỏ để Test
        )
        
        logging.info("[System] Rate limit safeguard engaged. Pausing execution for 5 seconds...")
        time.sleep(5)



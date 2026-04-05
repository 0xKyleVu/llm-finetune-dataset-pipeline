import os
import io
import tempfile
import logging
from minio import Minio
from minio.error import S3Error

# Đảm bảo symlinks trong thư mục HuggingFace không làm treo Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Engine bóc tách Document
from docling.document_converter import DocumentConverter

# Cấu hình Global Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CLOUD CONFIGURATION ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

SOURCE_BUCKET = "raw-data"
DEST_BUCKET = "processed-data"

def get_minio_client() -> Minio:
    """Bắt đầu kết nối đến Data Lake MinIO."""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )

def setup_minio_buckets(client: Minio):
    """Đảm bảo Bucket lưu trữ tồn tại."""
    try:
        if not client.bucket_exists(DEST_BUCKET):
            client.make_bucket(DEST_BUCKET)
            logging.info(f"[Initialization] Created destination bucket: s3://{DEST_BUCKET}")
        else:
            logging.info(f"[Initialization] Destination bucket already exists: s3://{DEST_BUCKET}")
    except S3Error as e:
        logging.error(f"[Initialization] MinIO server error: {e}")
        raise

def is_object_exists(client: Minio, bucket_name: str, object_name: str) -> bool:
    """Check sự tồn tại của file MinIO (Idempotency mapping)."""
    try:
        client.stat_object(bucket_name, object_name)
        return True
    except S3Error as e:
        if e.code == "NoSuchKey":
            return False
        raise

def parse_pdfs_to_markdown(client: Minio, converter=None):
    """
    Kéo các file PDF cục bộ về từ MinIO, tự động scan theo layout structure (Cột, Bảng, Tiêu đề), 
    rồi đẩy output chuẩn Markdown Format ngược về Data Lake MinIO.
    
    Args:
        client (Minio): Kết nối mạng MinIO.
        converter: Parameter hỗ trợ load model AI 1 lần duy nhất (lazy loading context).
    """
    try:
        pdf_objects = list(client.list_objects(SOURCE_BUCKET, prefix="pdf/"))
    except S3Error as e:
        logging.error(f"[Parser] Failed to list objects in '{SOURCE_BUCKET}': {e}")
        return

    if not pdf_objects:
        logging.warning(f"[Parser] No PDF files found in s3://{SOURCE_BUCKET}/pdf/")
        return
        
    logging.info(f"[Parser] Discovered {len(pdf_objects)} PDF documents. Initializing AI Engine...")
    
    # Init chậm: Chỉ load model 1 lần
    if converter is None:
        converter = DocumentConverter()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for obj in pdf_objects:
            object_name = obj.object_name
            file_name = os.path.basename(object_name)
            base_name = os.path.splitext(file_name)[0]
            
            output_md_object = f"markdown/{base_name}.md"
            
            if is_object_exists(client, DEST_BUCKET, output_md_object):
                logging.info(f"  -> [Idempotency] Markdown content exists, skipping: {output_md_object}")
                continue
                
            logging.info(f"[Parser] Analyzing document layout for: {file_name}")
            tmp_pdf_path = os.path.join(temp_dir, file_name)
            
            try:
                # Kéo stream dữ liệu của file PDF về Local tạm 
                client.fget_object(SOURCE_BUCKET, object_name, tmp_pdf_path)
                
                # Quét nhận diện tự động cấu trúc nội dung 
                result = converter.convert(tmp_pdf_path)
                
                # Biến đổi Serialization Markdown xuất Byte stream
                markdown_content = result.document.export_to_markdown()
                md_bytes = markdown_content.encode('utf-8')
                md_stream = io.BytesIO(md_bytes)
                
                client.put_object(
                    DEST_BUCKET,
                    output_md_object,
                    data=md_stream,
                    length=len(md_bytes),
                    content_type="text/markdown"
                )
                
                logging.info(f"  -> [Parser] Markdown Mapping successfully extracted. Uploaded to: s3://{DEST_BUCKET}/{output_md_object}")
                
            except Exception as e:
                logging.error(f"[Parser] Critical error occurred parsing document '{file_name}': {e}")
            finally:
                if os.path.exists(tmp_pdf_path):
                    os.remove(tmp_pdf_path)

if __name__ == "__main__":
    logging.info("=========================================")
    logging.info("[System] STARTING PROCESSING LAYER")
    logging.info("=========================================")
    
    minio_client = get_minio_client()
    setup_minio_buckets(minio_client)
    
    parse_pdfs_to_markdown(client=minio_client)
    
    logging.info("[System] Document Parser pipeline execution completed successfully.")

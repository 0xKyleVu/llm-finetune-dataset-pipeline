import os
import json
import logging
import io
import time
from dagster import sensor, RunRequest, AssetSelection, SensorEvaluationContext
from src.ingestion.arxiv_crawler import get_minio_client, setup_minio_buckets

# --- CONFIGURATION ---
HOT_FOLDER = "/app/hot_folder"
RAW_BUCKET = "raw-data"

@sensor(target=AssetSelection.all() - AssetSelection.assets("arxiv_raw_data"), minimum_interval_seconds=30)
def hot_folder_sensor(context: SensorEvaluationContext):
    """
    Theo dõi thư mục hot_folder. Khi có file PDF mới:
    1. Ingest vào MinIO.
    2. Kích hoạt Pipeline từ bước Parser trở đi.
    """
    if not os.path.exists(HOT_FOLDER):
        os.makedirs(HOT_FOLDER)

    client = get_minio_client()
    setup_minio_buckets(client)

    new_files = [f for f in os.listdir(HOT_FOLDER) if f.lower().endswith(".pdf")]
    
    for file_name in new_files:
        file_path = os.path.join(HOT_FOLDER, file_name)
        paper_id = os.path.splitext(file_name)[0]
        
        context.log.info(f"✨ [Sensor] New paper detected: {file_name}")
        
        try:
            # 1. Upload PDF
            with open(file_path, "rb") as f:
                client.put_object(
                    RAW_BUCKET,
                    f"pdf/{file_name}",
                    data=f,
                    length=os.path.getsize(file_path),
                    content_type="application/pdf"
                )
            context.log.info(f"  -> Uploaded PDF to s3://{RAW_BUCKET}/pdf/{file_name}")
            # 2. Tạo Metadata
            metadata = {
                "id": paper_id,
                "title": paper_id.replace("_", " ").replace("-", " "),
                "summary": "Manual identification: Source file ingested via Dagster Sensor.",
                "authors": "Local Ingest",
                "category": "manual_ingest"
            }
            meta_json = json.dumps(metadata).encode('utf-8')
            client.put_object(
                RAW_BUCKET,
                f"metadata/{paper_id}.json",
                data=io.BytesIO(meta_json),
                length=len(meta_json),
                content_type="application/json"
            )
            context.log.info(f"  -> Created metadata for: {paper_id}")

            # 3. Xóa file gốc (Cleanup)
            os.remove(file_path)
            context.log.info(f"  -> [Cleanup] Removed local file: {file_name}")

            # 4. Trả về RunRequest (Chỉ chạy từ Parser trở đi)
            yield RunRequest(
                run_key=f"ingest_{paper_id}_{int(time.time())}",
                tags={"source": "hot_folder", "paper_id": paper_id}
            )

        except Exception as e:
            context.log.error(f"[Sensor] Failed to ingest {file_name}: {e}")

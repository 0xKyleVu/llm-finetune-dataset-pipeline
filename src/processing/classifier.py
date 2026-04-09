import os
import json
import logging
import io
from minio import Minio
from minio.error import S3Error
from collections import defaultdict
from transformers import pipeline

# Cấu hình Global Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CLOUD CONFIGURATION ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

SOURCE_BUCKET = "cleaned-data"
DEST_BUCKET = "classified-data"

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

def classify_chunks_layer(client: Minio):
    """
    Kích hoạt phân loại phân luồng Zero-shot semantic lấy JSON block mảng từ MinIO.
    Sau khi phân phối dán nhãn, output sẽ auto upload đổ về đích.
    """
    try:
        chunk_objects = list(client.list_objects(SOURCE_BUCKET, prefix="chunks/"))
    except S3Error as e:
        logging.error(f"[Classifier] Failed to list objects in '{SOURCE_BUCKET}': {e}")
        return

    if not chunk_objects:
        logging.warning(f"[Classifier] No JSON array files found in s3://{SOURCE_BUCKET}/chunks/")
        return

    # Khai báo các nhãn Candidate Labels cho quá trình Zero-shot inference
    candidate_labels = [
        "artificial intelligence",
        "machine learning",
        "cryptography",
        "cryptocurrency",
        "blockchain",
        "cybersecurity",
        "information theory",
        "systems engineering",
        "mathematics",
        "healthcare",
        "education",
        "quantitative finance",
        "computer networks",
        "signal processing",
        "military",
        "metadata and references"
    ]

    logging.info("[Classifier] Initializing Zero-shot AI Pipeline (HuggingFace)...")
    logging.info("[Classifier] Selected model architecture: 'facebook/bart-large-mnli'")
    
    # Auto-detect CPU/GPU
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except ImportError:
        device = -1
    logging.info(f"[Classifier] Hardware accelerator deployed: {'GPU (CUDA)' if device == 0 else 'CPU'}")
    
    classifier = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli",
        device=device
    )
    logging.info("[Classifier] AI Inference pipeline loaded successfully.")

    for obj in chunk_objects:
        object_name = obj.object_name
        file_name = os.path.basename(object_name)
        output_object_name = f"chunks/{file_name}"
        
        if is_object_exists(client, DEST_BUCKET, output_object_name):
            logging.info(f"  -> [Idempotency] Labeled array exists, skipping: {output_object_name}")
            continue

        logging.info(f"[Classifier] Executing semantic inference for document: {file_name}")
        try:
            # Gắp Structured context kéo về RAM xử lý in-memory
            response = client.get_object(SOURCE_BUCKET, object_name)
            chunks = json.loads(response.read().decode('utf-8'))
            response.close()
            response.release_conn()
            
            # Pre-filter dọn dẹp các chunk quá ngắn trước khi ném vào Execution
            valid_chunks = []
            valid_contents = []
            for chunk in chunks:
                content = chunk.get("content", "")
                if content and len(content) >= 50:
                    valid_chunks.append(chunk)
                    valid_contents.append(content)
            
            if not valid_contents:
                logging.warning(f"  -> [Filter] No logically coherent chunks found in {file_name}")
                continue
            
            total_chunks = len(valid_contents)
            logging.info(f"  -> Valid chunks: {total_chunks}. Commencing batch processing sequence...")
            
            BATCH_SIZE = 16
            classified_chunks = []
            
            for batch_start in range(0, total_chunks, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_chunks)
                batch_contents = valid_contents[batch_start:batch_end]
                batch_chunks = valid_chunks[batch_start:batch_end]
                
                results = classifier(
                    batch_contents,
                    candidate_labels,
                    multi_label=True,
                    hypothesis_template="This text is about {}.",
                    batch_size=BATCH_SIZE
                )
                
                if isinstance(results, dict):
                    results = [results]
                
                for chunk, result in zip(batch_chunks, results):
                    top_label = result['labels'][0]
                    top_score = result['scores'][0]
                    
                    if top_score < 0.45:
                        top_label = "metadata and references"
                        quality_tier = "Low"
                    elif top_score >= 0.7:
                        quality_tier = "High"
                    else:
                        quality_tier = "Medium"
                    
                    chunk["category"] = top_label
                    chunk["confidence_score"] = round(top_score, 4)
                    chunk["quality_tier"] = quality_tier
                    classified_chunks.append(chunk)
                
                logging.info(f"  -> [Inference] Batch segment {batch_start+1}-{batch_end}/{total_chunks} processed.")

            # Upload annotated payload vào MinIO
            json_bytes = json.dumps(classified_chunks, ensure_ascii=False, indent=2).encode('utf-8')
            json_stream = io.BytesIO(json_bytes)
            
            client.put_object(
                DEST_BUCKET,
                output_object_name,
                data=json_stream,
                length=len(json_bytes),
                content_type="application/json"
            )

            logging.info(f"  -> [Classifier] Classification accomplished. Linked {len(classified_chunks)} chunks to s3://{DEST_BUCKET}/{output_object_name}")

        except Exception as e:
            logging.error(f"[Classifier] Critical inference failure for document '{file_name}': {e}")

    # Sau khi phân loại xong toàn bộ, trả về báo cáo thống kê
    return generate_classification_report(client)


def generate_classification_report(client: Minio) -> dict:
    """
    Quét toàn bộ dữ liệu đã phân loại trong MinIO và trả về báo cáo thống kê.

    Returns:
        dict: Chứa category_counts, tier_counts, confidence_by_label,
              total_chunks, usable_pct, overall_conf, status.
              Trả về None nếu không có dữ liệu.

    Cách đọc kết quả:
    - High Tier > 60%: Hệ thống đang hoạt động tốt, dữ liệu rất sạch.
    - Avg Confidence > 0.7: Model có độ chắc chắn cao, kết quả đáng tin cậy.
    - Avg Confidence < 0.5 cho một nhãn: Nhãn đó có thể bị nhầm lẫn, cần xem xét lại.
    """
    logging.info("\n" + "=" * 60)
    logging.info("  CLASSIFICATION REPORT")
    logging.info("=" * 60)

    try:
        obj_list = list(client.list_objects(DEST_BUCKET, prefix="chunks/"))
    except S3Error as e:
        logging.error(f"[Report] Failed to access classified bucket: {e}")
        return

    if not obj_list:
        logging.warning("[Report] No classified data found.")
        return

    # Thu thập dữ liệu từ tất cả các file đã phân loại
    category_counts = defaultdict(int)
    tier_counts = defaultdict(int)
    # confidence_scores[label] = [score1, score2, ...]
    confidence_by_label = defaultdict(list)
    total_chunks = 0

    for obj in obj_list:
        try:
            response = client.get_object(DEST_BUCKET, obj.object_name)
            chunks = json.loads(response.read().decode('utf-8'))
            response.close()
            response.release_conn()

            for chunk in chunks:
                category = chunk.get("category", "unknown")
                tier = chunk.get("quality_tier", "unknown")
                score = chunk.get("confidence_score", 0.0)

                category_counts[category] += 1
                tier_counts[tier] += 1
                confidence_by_label[category].append(score)
                total_chunks += 1

        except Exception as e:
            logging.warning(f"[Report] Could not read {obj.object_name}: {e}")
            continue

    if total_chunks == 0:
        logging.warning("[Report] No chunks found in classified data.")
        return

    # --- PHẦN 1: Phân bổ chủ đề ---
    logging.info(f"\n  [1/3] CATEGORY DISTRIBUTION (Total: {total_chunks} chunks)")
    logging.info("  " + "-" * 50)
    for label, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / total_chunks * 100
        bar = "█" * int(pct / 2)  # Thanh tiến trình ASCII
        avg_conf = sum(confidence_by_label[label]) / len(confidence_by_label[label])
        logging.info(f"  {label:<30} {count:>5} ({pct:5.1f}%)  conf_avg: {avg_conf:.3f}  {bar}")

    # --- PHẦN 2: Phân bổ Quality Tier ---
    logging.info(f"\n  [2/3] QUALITY TIER DISTRIBUTION")
    logging.info("  " + "-" * 50)
    tier_order = ["High", "Medium", "Low"]
    for tier in tier_order:
        count = tier_counts.get(tier, 0)
        pct = count / total_chunks * 100
        bar = "█" * int(pct / 2)
        logging.info(f"  {tier:<10} {count:>5} ({pct:5.1f}%)  {bar}")

    usable_pct = (tier_counts.get("High", 0) + tier_counts.get("Medium", 0)) / total_chunks * 100
    logging.info(f"\n  => Usable data (High + Medium): {usable_pct:.1f}%")

    # --- PHẦN 3: Đánh giá sức khỏe ---
    overall_conf = sum(s for scores in confidence_by_label.values() for s in scores) / total_chunks
    logging.info(f"\n  [3/3] OVERALL CONFIDENCE SCORE: {overall_conf:.4f}")
    if overall_conf >= 0.7:
        logging.info("  => STATUS: EXCELLENT - Model is highly confident in classifications.")
    elif overall_conf >= 0.5:
        logging.info("  => STATUS: ACCEPTABLE - Some ambiguous chunks exist, review 'Low' tier.")
    else:
        logging.info("  => STATUS: WARNING - Low confidence. Consider refining candidate labels.")

    logging.info("\n" + "=" * 60 + "\n")

    # Trả về dữ liệu thống kê để Dagster asset có thể hiển thị trên UI
    if overall_conf >= 0.7:
        status = "EXCELLENT"
    elif overall_conf >= 0.5:
        status = "ACCEPTABLE"
    else:
        status = "WARNING"

    return {
        "total_chunks": total_chunks,
        "category_counts": dict(category_counts),
        "tier_counts": dict(tier_counts),
        "confidence_by_label": {k: round(sum(v)/len(v), 4) for k, v in confidence_by_label.items()},
        "usable_pct": round(usable_pct, 2),
        "overall_conf": round(overall_conf, 4),
        "status": status,
    }

if __name__ == "__main__":
    logging.info("=========================================")
    logging.info("[System] STARTING CLASSIFICATION LAYER")
    logging.info("=========================================")
    
    minio_client = get_minio_client()
    setup_minio_buckets(minio_client)
    
    classify_chunks_layer(client=minio_client)
    
    logging.info("[System] Classification pipeline execution completed.")

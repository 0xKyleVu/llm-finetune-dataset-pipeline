import sys
import os
import re
import random
import asyncio
import duckdb
import json
import logging
import io
from pydantic import BaseModel
from ollama import AsyncClient
from minio import Minio
from minio.error import S3Error

sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình Global Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FinetuneSample(BaseModel):
    instruction: str
    input: str
    output: str

# --- CLOUD CONFIGURATION ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

MODEL_NAME = "llama3.2"
SOURCE_BUCKET = "classified-data"
DEST_BUCKET = "finetune-data"
OUTPUT_OBJECT = "dataset/finetune_dataset.jsonl"
MIN_TEXT_LENGTH = 200
MAX_CONCURRENT = 2

# Top-level dạng regex nhận diện rác academic 
_CITATION_MARKERS = re.compile(
    r'\b(pp\.|vol\.|no\.|In:|arXiv:\d{4}\.\d+|doi\.org|ISBN|ISSN|Proc\.|Conf\.|Adv\.|Trans\.|Springer|Elsevier|Wiley)\b'
    r'|In Proceedings\b'
    r'|\[Online\]\.?\s*Available'
    r'|\((19|20)\d{2}\)'
    r'|,\s*(19|20)\d{2}[\.\s]'
    r'|\.\s*(19|20)\d{2}\.'
)
_AFFILIATION_SIGNALS = re.compile(
    r'(University|Institute|Lab|Research|Department|Corp|Inc\.|@[a-zA-Z])',
    re.IGNORECASE
)
_EMAIL = re.compile(r'[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}')
_COPYRIGHT = re.compile(
    r'©|\bAll rights reserved\b|\bpermission\b.{0,40}\bIEEE\b'
    r'|\bAccepted at\b|\bco-located with\b|\bpersonal use\b',
    re.IGNORECASE
)

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

def is_noise_chunk(text: str) -> bool:
    """
    Bắn về True nếu text block đc cho rác semantic (bibliography/copyright/authors).
    Bảo toàn nếu chứa methodology đan xen name.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return True

    # Tier 0: Copyright & Legal constraints
    if _COPYRIGHT.search(text):
        return True

    # Tier 1: Heavy Bibliography
    bullet_lines = [l for l in lines if l.startswith('- ')]
    bullet_ratio = len(bullet_lines) / len(lines) if lines else 0
    if bullet_ratio > 0.5 and bullet_lines:
        citation_bullets = sum(1 for l in bullet_lines if _CITATION_MARKERS.search(l))
        if citation_bullets / len(bullet_lines) >= 0.5:
            return True

    # Tier 2: Author Affiliations (Tên tác giả, phòng ban liên đới)
    if len(lines) <= 6:
        affil_lines = sum(
            1 for l in lines
            if _AFFILIATION_SIGNALS.search(l) or _EMAIL.search(l)
        )
        if lines and affil_lines / len(lines) >= 0.5:
            return True

    # Tier 3: Markup tần suất khóa academic
    bib_count = len(_CITATION_MARKERS.findall(text))
    word_count = len(text.split())
    if word_count > 0 and bib_count / word_count > 0.04:
        return True

    return False

async def process_single_row(client, semaphore, idx, row, total):
    """Lệnh Conversational prompt payload per row kèm Semaphore."""
    content = row['clean_text']

    # Filter 1: Lọc bảng số (>40% digits overflow)
    digit_count = sum(c.isdigit() for c in content)
    if digit_count / len(content) > 0.4:
        return None

    # Filter 2: Lọc rác academic
    if is_noise_chunk(content):
        return None

    is_conceptual = random.random() < 0.7
    question_type = "CONCEPTUAL" if is_conceptual else "TECHNICAL"

    system_prompt = (
        "You are a Senior Research Scientist specializing in technical papers. "
        "Your goal is to synthesize high-quality training data for LLM fine-tuning. "
        "The 'instruction' must be a natural, human-like question about the EXCERPT content. "
        "The 'output' must be grounded STRICTLY in the provided TECHNICAL EXCERPT. "
        "CRITICAL RULES: "
        "(1) DO NOT invent, assume, or hallucinate any numbers, results, or claims not present in the excerpt. "
        "(2) If specific data is absent, explain the concept without fabricating figures. "
        "(3) DO NOT write or reproduce mathematical formulas unless they appear verbatim in the excerpt. "
        "(4) The 'output' must be at least 2 complete sentences. "
        "(5) DO NOT use phrases like 'According to the text' or 'The text states'."
    )

    if is_conceptual:
        task_instruction = "Ask a conceptual or explanatory question about a methodology, problem, or theory."
    else:
        task_instruction = "Ask a specific technical question about parameters, algorithms, results, or data points."

    user_prompt = f"""
PAPER TITLE: {row['title']}
PAPER SUMMARY: {row['summary']}

TECHNICAL EXCERPT FROM THE PAPER:
{content}

TASK: {task_instruction}
FORMAT: Output a single JSON with 'instruction' (the question) and 'output' (the expert answer).
RULES: English only. Strictly based on the excerpt and paper context.
"""

    async with semaphore:
        logging.info(f"[Generator] Instructing sequence [{idx}/{total}] [{question_type}]: {row['category']}")
        try:
            response = await client.chat(
                model=MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                format=FinetuneSample.model_json_schema(),
                options={'temperature': 0.3}
            )

            raw_response = json.loads(response.message.content)

            instruction = raw_response.get("instruction") or raw_response.get("question") or raw_response.get("Q")
            output = raw_response.get("output") or raw_response.get("answer") or raw_response.get("A")

            # Post-filter: Lọc Instruction
            if not instruction or len(instruction) < 20 or not output or len(output) < 80:
                return None

            if "SKIP" in instruction.upper() or "SKIP" in output.upper():
                return None

            # Track Copy-paste plagiarism từ AI Generator
            out_words = set(output.lower().split())
            txt_words = set(content.lower().split())
            if out_words and txt_words:
                overlap = len(out_words & txt_words) / len(out_words)
                if overlap > 0.85:
                    return None

            logging.info(f"  -> [Verified] Generated logic: {instruction[:60]}...")
            return {
                "source_paper_id": row['paper_id'],
                "clean_text": content,
                "category": row['category'],
                "finetune_sample": {
                    "instruction": instruction,
                    "input": "",
                    "output": output
                }
            }

        except Exception as e:
            logging.error(f"[Generator] Execution fault at process row {idx}: {e}")
            return None

async def generate_finetune_data():
    """
    Hệ thống Orchestrator trung tâm nối các cục LLM API cắm vào Cloud-Native Data Bucket thẳng tắp.
    """
    logging.info("[Initialization] Establishing DuckDB HTTPFS engine...")
    con = duckdb.connect(':memory:')
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")

    # Setup DuckDB sử dụng MinIO credentials 
    s3_use_ssl = "true" if MINIO_SECURE else "false"
    con.execute(f"""
        SET s3_endpoint='{MINIO_ENDPOINT}';
        SET s3_access_key_id='{MINIO_ACCESS_KEY}';
        SET s3_secret_access_key='{MINIO_SECRET_KEY}';
        SET s3_url_style='path';
        SET s3_use_ssl={s3_use_ssl};
    """)

    logging.info(f"[DB] Executing distributed HTTPFS query targeting s3://{SOURCE_BUCKET}...")

    query = f"""
        SELECT 
            c.content as clean_text,
            c.category as category,
            c.paper_id as paper_id,
            m.title as title,
            m.summary as summary
        FROM read_json_auto('s3://{SOURCE_BUCKET}/chunks/*.json') c
        JOIN read_json_auto('s3://raw-data/metadata/*.json') m
          ON c.paper_id = m.id
        WHERE c.quality_tier IN ('High', 'Medium') 
        AND length(c.content) > {MIN_TEXT_LENGTH}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY c.content ORDER BY c.quality_tier) = 1
        ORDER BY c.category, length(c.content) DESC
        LIMIT 500
    """

    try:
        rows = con.execute(query).df().to_dict('records')
        logging.info(f"[DB] Retrieved {len(rows)} qualified records combined with architectural metadata.")
    except Exception as e:
        logging.error(f"[DB] Data query operation denied from MinIO: {e}")
        return

    logging.info(f"[Generator] Initializing LLM cluster ({MODEL_NAME}) with concurrency level: {MAX_CONCURRENT}")
    
    # Lấy địa chỉ Ollama từ biến môi trường (mặc định là localhost nếu chạy ngoài Docker)
    ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    logging.info(f"[Generator] Connecting to Ollama at: {ollama_url}")
    
    ollama_client = AsyncClient(host=ollama_url)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        process_single_row(ollama_client, semaphore, idx, row, len(rows))
        for idx, row in enumerate(rows, 1)
    ]
    results = await asyncio.gather(*tasks)

    # Filter các dòng pure rỗng ko/ít ý nghĩa
    records = [r for r in results if r is not None]
    
    if not records:
        logging.warning("[Generator] Synchronization yielded zero valid synthetic sequences. Process terminated.")
        return

    logging.info(f"[Pipeline] Acquired {len(records)} robust sequences. Finalizing cloud deployment...")
    
    # Marshal outputs thành luồng Memory-bound JSONL buffer
    jsonl_output = "\n".join([json.dumps(record, ensure_ascii=False) for record in records])
    jsonl_bytes = jsonl_output.encode('utf-8')
    jsonl_stream = io.BytesIO(jsonl_bytes)
    
    minio_client = get_minio_client()
    setup_minio_buckets(minio_client)
    
    try:
        minio_client.put_object(
            DEST_BUCKET,
            OUTPUT_OBJECT,
            data=jsonl_stream,
            length=len(jsonl_bytes),
            content_type="application/jsonlines"
        )
        logging.info(f"[IO] JSONL fine-tuning dataset deployed successfully to s3://{DEST_BUCKET}/{OUTPUT_OBJECT}")
    except Exception as e:
        logging.error(f"[IO] Buffer flush failed during MinIO upload sequence: {e}")

if __name__ == "__main__":
    logging.info("=========================================")
    logging.info("[System] STARTING DATASET GENERATOR")
    logging.info("=========================================")
    asyncio.run(generate_finetune_data())
    logging.info("[System] Synthesis pipeline execution completed.")

import sys
import os
import re
import random
import asyncio
import duckdb
import json
from pydantic import BaseModel
from ollama import AsyncClient

sys.stdout.reconfigure(encoding='utf-8')

class FinetuneSample(BaseModel):
    instruction: str
    input: str
    output: str

# Cấu hình (Configuration)
MODEL_NAME = "llama3.2"
BUCKET_NAME = "classified-data"
OUTPUT_FILE = "test_data/finetune_dataset.jsonl"
MIN_TEXT_LENGTH = 200
MAX_CONCURRENT = 2

# Ký hiệu Bibliography
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


def is_noise_chunk(text: str) -> bool:
    """
    Trả về True nếu chunk là Rác (bibliography / danh sách tác giả / copyright).
    GIỮ LẠI các đoạn có tên tác giả nhưng kèm phát hiện / phương pháp giá trị.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return True

    # Tầng 0: Copyright / Legal Notice
    if _COPYRIGHT.search(text):
        return True

    # Tầng 1: Bibliography / Reference List
    bullet_lines = [l for l in lines if l.startswith('- ')]
    bullet_ratio = len(bullet_lines) / len(lines)
    if bullet_ratio > 0.5 and bullet_lines:
        citation_bullets = sum(1 for l in bullet_lines if _CITATION_MARKERS.search(l))
        if citation_bullets / len(bullet_lines) >= 0.5:
            return True

    # Tầng 2: Author Affiliation Block
    if len(lines) <= 6:
        affil_lines = sum(
            1 for l in lines
            if _AFFILIATION_SIGNALS.search(l) or _EMAIL.search(l)
        )
        if affil_lines / len(lines) >= 0.5:
            return True

    # Tầng 3: Mật độ ký hiệu học thuật quá cao
    bib_count = len(_CITATION_MARKERS.findall(text))
    word_count = len(text.split())
    if word_count > 0 and bib_count / word_count > 0.04:
        return True

    return False


async def process_single_row(client, semaphore, idx, row, total):
    """Xử lý 1 row với semaphore giới hạn concurrency."""
    content = row['clean_text']

    # Lọc 1: Bảng biểu rác (>40% chữ số)
    digit_count = sum(c.isdigit() for c in content)
    if digit_count / len(content) > 0.4:
        return None

    # Lọc 2: Bibliography / danh sách tác giả
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
        print(f"--- [{idx}/{total}] [{question_type}]: {row['category']}")
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

            # Bỏ qua output quá ngắn (< 80 ký tự)
            if not instruction or len(instruction) < 20 or not output or len(output) < 80:
                return None

            if "SKIP" in instruction.upper() or "SKIP" in output.upper():
                return None

            # Kiểm tra copy overlap
            out_words = set(output.lower().split())
            txt_words = set(content.lower().split())
            if out_words and txt_words:
                overlap = len(out_words & txt_words) / len(out_words)
                if overlap > 0.85:
                    return None

            print(f"  [\u2713] {instruction[:60]}...")
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
            print(f"[ERROR] Dòng {idx}: {e}")
            return None


async def generate_finetune_data():
    # 1. Khởi tạo DuckDB
    print("[▶] Khởi tạo DuckDB...")
    con = duckdb.connect(':memory:')
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")

    # Set env MinIO
    con.execute("""
        SET s3_endpoint='localhost:9000';
        SET s3_access_key_id='minioadmin';
        SET s3_secret_access_key='minioadmin';
        SET s3_url_style='path';
        SET s3_use_ssl=false;
    """)

    # 2. Query từ MinIO
    print("[INFO] Đang truy vấn dữ liệu từ MinIO...")

    query = f"""
        SELECT 
            c.content as clean_text,
            c.category as category,
            c.paper_id as paper_id,
            m.title as title,
            m.summary as summary
        FROM read_json_auto('s3://{BUCKET_NAME}/chunks/*.json') c
        JOIN read_json_auto('s3://raw-data/metadata/*.json') m
          ON c.paper_id = m.id
        WHERE c.quality_tier IN ('High', 'Medium') 
        AND length(c.content) > {MIN_TEXT_LENGTH}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY c.content ORDER BY c.quality_tier) = 1
        ORDER BY c.category, length(c.content) DESC
        LIMIT 300
    """

    try:
        rows = con.execute(query).df().to_dict('records')
        print(f"[INFO] Đã tải {len(rows)} record (High & Medium) kèm Metadata.")
    except Exception as e:
        print(f"[ERROR] Không thể lấy dữ liệu: {e}")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"[INFO] Bắt đầu sinh dataset với {MODEL_NAME} (concurrency={MAX_CONCURRENT})")

    # 3. Async generation với semaphore
    client = AsyncClient(host='http://localhost:11434')
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        process_single_row(client, semaphore, idx, row, len(rows))
        for idx, row in enumerate(rows, 1)
    ]
    results = await asyncio.gather(*tasks)

    # 4. Ghi kết quả (chỉ records thành công)
    records = [r for r in results if r is not None]
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for record in records:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\n[SUCCESS] Hoàn tất: {len(records)} records.")
    print("=======================================================")

if __name__ == "__main__":
    asyncio.run(generate_finetune_data())

import os
import glob
import json
import logging
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_classified_directories(base_dir: str = 'test_data'):
    """Tạo thư mục cho dữ liệu đã được gán nhãn (Classified Data)"""
    classified_dir = os.path.join(base_dir, 'classified', 'chunks')
    os.makedirs(classified_dir, exist_ok=True)
    return classified_dir

def classify_chunks_layer(input_dir: str, output_dir: str):
    """
    Sử dụng Zero-shot Classification để phân loại nội dung các chunks.
    """
    chunk_files = glob.glob(os.path.join(input_dir, '*.json'))
    if not chunk_files:
        logging.warning(f"Không tìm thấy file nào trong {input_dir}")
        return

    # Candidate Labels cho Zero-shot
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

    logging.info("Đang khởi tạo AI Pipeline (HuggingFace Zero-shot)...")
    logging.info("Mô hình được chọn: 'facebook/bart-large-mnli'")
    
    logging.info("Đang khởi tạo mô hình Zero-shot: facebook/bart-large-mnli")
    classifier = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli",
        device=0
    )
    logging.info("Khởi tạo mô hình thành công.")

    for file_path in chunk_files:
        file_name = os.path.basename(file_path)
        output_file_path = os.path.join(output_dir, file_name)
        
        if os.path.exists(output_file_path):
            logging.info(f"Bỏ qua file đã xử lý: {file_name}")
            continue

        logging.info(f"Đang xử lý: {file_name}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            classified_chunks = []
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks):
                content = chunk.get("content", "")
                if not content or len(content) < 50:
                    continue
                
                result = classifier(
                    content, 
                    candidate_labels,
                    multi_label=True,
                    hypothesis_template="This text is about {}."
                )
                
                top_label = result['labels'][0]
                top_score = result['scores'][0]
                
                # Phân tầng chất lượng dữ liệu
                # Ngưỡng tối thiểu 0.45 để giảm nhiễu (Dưới ngưỡng này coi là Metadata/Rác)
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
                
                if (i + 1) % 10 == 0 or (i + 1) == total_chunks:
                    logging.info(f" Tiến độ: {i + 1}/{total_chunks} chunks | [{top_label}] ({top_score*100:.1f}%)")

            # Ghi ra file JSON
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(classified_chunks, f, ensure_ascii=False, indent=2)

            logging.info(f"Hoàn tất: {file_name} ({len(classified_chunks)} chunks).")

        except Exception as e:
            logging.error(f"Lỗi xử lý file {file_name}: {e}")

if __name__ == "__main__":
    test_base_dir = 'test_data'
    input_cleaned_dir = os.path.join(test_base_dir, 'cleaned', 'chunks')
    output_classified_dir = setup_classified_directories(test_base_dir)
    
    logging.info("BẮT ĐẦU LAYER PHÂN LOẠI (CLASSIFICATION)")
    classify_chunks_layer(input_dir=input_cleaned_dir, output_dir=output_classified_dir)
    logging.info("Layer Phân loại kết thúc.")

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

    # Danh sách 10 thẻ hạng mục chuẩn
    candidate_labels = [
        "Artificial Intelligence",
        "Mathematics",
        "Finance",
        "Military",
        "Hardware",
        "Physics",
        "Healthcare",
        "Education",
        "Software Engineering",
        "Metadata and References"
    ]

    logging.info("Đang khởi tạo AI Pipeline (HuggingFace Zero-shot)...")
    logging.info("Mô hình được chọn: 'facebook/bart-large-mnli'")
    
    # Sử dụng bart-large-mnli
    classifier = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli",
        device=0 # Chạy bằng GPU
    )
    logging.info("Khởi tạo AI thành công!")

    for file_path in chunk_files:
        file_name = os.path.basename(file_path)
        output_file_path = os.path.join(output_dir, file_name)
        
        # Idempotent
        if os.path.exists(output_file_path):
            logging.info(f"File đã được phân loại trước đó, bỏ qua: {file_name}")
            continue

        logging.info(f"\n[TIẾN TRÌNH] Đang phân loại file: {file_name}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            classified_chunks = []
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks):
                content = chunk.get("content", "")
                if not content or len(content) < 50:
                    continue # Bỏ qua rác
                
                # Gọi AI đọc và chấm điểm đa nhãn độc lập (Multi-label)
                result = classifier(
                    content, 
                    candidate_labels,
                    multi_label=True,
                    hypothesis_template="The topic of this text is {}."
                )
                
                # Cắt lấy Kết quả Top 1
                top_label = result['labels'][0]
                top_score = result['scores'][0]
                
                chunk["category"] = top_label
                chunk["confidence_score"] = round(top_score, 4)
                
                classified_chunks.append(chunk)
                
                # Log tiến độ để user khỏi sốt ruột
                if (i + 1) % 5 == 0 or (i + 1) == total_chunks:
                    logging.info(f"  -> Đã gán nhãn {i + 1}/{total_chunks} chunks. Gần nhất: [{top_label}] ({top_score*100:.1f}%)")

            # Ghi ra file JSON chuẩn
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(classified_chunks, f, ensure_ascii=False, indent=2)

            logging.info(f"Hoàn tất file {file_name} với {len(classified_chunks)} chunks có nhãn. Lưu tại: {output_file_path}")

        except Exception as e:
            logging.error(f"Lỗi khi xử lý file {file_name}: {e}")

if __name__ == "__main__":
    test_base_dir = '../test_data' if not os.path.exists('test_data') else 'test_data'
    
    # Lấy đầu vào từ output của cleaner
    input_cleaned_dir = os.path.join(test_base_dir, 'cleaned', 'chunks')
    output_classified_dir = setup_classified_directories(test_base_dir)
    
    logging.info("\n=========================================\nBẮT ĐẦU LAYER PHÂN LOẠI (CLASSIFICATION)\n=========================================")
    classify_chunks_layer(input_dir=input_cleaned_dir, output_dir=output_classified_dir)
    logging.info("Layer Phân loại kết thúc thành công!")

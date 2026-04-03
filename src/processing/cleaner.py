import os
import glob
import json
import logging
import re
import hashlib

# Import ftfy: fixes text for you
import ftfy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_cleaned_directories(base_dir: str = 'test_data'):
    """Tạo thư mục cho dữ liệu đã được làm sạch (Cleaned Data)"""
    cleaned_dir = os.path.join(base_dir, 'cleaned', 'chunks')
    os.makedirs(cleaned_dir, exist_ok=True)
    return cleaned_dir

def clean_text(text: str) -> str:
    """
    Thực hiện cleaning văn bản cho LLM.
    """
    if not text:
        return ""
        
    # 1. Fix lỗi Unicode 
    text = ftfy.fix_text(text)
    
    # 2. Xóa Citation không cần thiết
    # Ví dụ: loại bỏ [12], [3, 4], [1-5]
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    text = re.sub(r'\[\d+(?:-\d+)*\]', '', text)
    
    # 3. Xóa URL web dài dòng (giữ lại DOI vì là metadata quan trọng)
    text = re.sub(r'https?://(?!doi\.org)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # 4. Dọn whitespace
    text = re.sub(r'\n{3,}', '\n\n', text) # Nén 3 dòng trống liên tiếp thành 2 dòng
    text = re.sub(r' {2,}', ' ', text)     # Ép 2 khoảng trắng liên tiếp làm 1
    
    return text.strip()

def clean_chunks_layer(input_dir: str, output_dir: str):
    """
    Đọc các file chunk, làm sạch từng khối và xuất ra thư mục mới.
    """
    chunk_files = glob.glob(os.path.join(input_dir, '*.json'))
    if not chunk_files:
        logging.warning(f"Không tìm thấy khối dữ liệu nào trong {input_dir}")
        return

    logging.info(f"Phát hiện {len(chunk_files)} file Chunks. Khởi động Cleaner...")

    for file_path in chunk_files:
        file_name = os.path.basename(file_path)
        output_file_path = os.path.join(output_dir, file_name)
        
        # Idempotent
        if os.path.exists(output_file_path):
            logging.info(f"File đã được làm sạch trước đó, bỏ qua: {file_name}")
            continue

        logging.info(f"Cleaning: {file_name}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            cleaned_chunks = []
            seen_hashes = set()  # Dedup bằng content hash
            dup_count = 0
            
            for chunk in chunks:
                original_content = chunk.get("content", "")
                sanitized_content = clean_text(original_content)
                
                # Bỏ qua các khối rỗng sau khi đã làm sạch
                if not sanitized_content:
                    continue
                
                # Dedup: bỏ qua chunk có nội dung trùng lặp
                content_hash = hashlib.md5(sanitized_content.encode()).hexdigest()
                if content_hash in seen_hashes:
                    dup_count += 1
                    continue
                seen_hashes.add(content_hash)
                    
                # Ghi đè chữ sạch vào nội dung
                chunk["content"] = sanitized_content
                chunk["char_length"] = len(sanitized_content)
                cleaned_chunks.append(chunk)
            
            if dup_count > 0:
                logging.info(f"  Đã loại {dup_count} chunk trùng lặp.")

            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_chunks, f, ensure_ascii=False, indent=2)

            logging.info(f"Làm sạch thành công {len(cleaned_chunks)} khối văn bản. Lưu tại: {output_file_path}")

        except Exception as e:
            logging.error(f"Lỗi khi làm sạch file {file_name}: {e}")

if __name__ == "__main__":
    test_base_dir = '../test_data' if not os.path.exists('test_data') else 'test_data'
    
    input_chunks_dir = os.path.join(test_base_dir, 'processed', 'chunks')
    output_cleaned_dir = setup_cleaned_directories(test_base_dir)
    
    logging.info("\n=========================================\nBẮT ĐẦU LAYER LÀM SẠCH (CLEANING)\n=========================================")
    clean_chunks_layer(input_dir=input_chunks_dir, output_dir=output_cleaned_dir)
    logging.info("Hoàn tất Cleaning!")

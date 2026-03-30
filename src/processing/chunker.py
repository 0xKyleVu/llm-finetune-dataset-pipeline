import os
import glob
import json
import logging
from typing import List, Dict

# Import Langchain
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_chunk_directories(base_dir: str = 'test_data'):
    """Tạo thư mục để chứa đống Text đã được băm nhỏ"""
    chunked_dir = os.path.join(base_dir, 'processed', 'chunks')
    os.makedirs(chunked_dir, exist_ok=True)
    return chunked_dir

def chunk_markdown_files(input_dir: str, output_dir: str):
    """
    Sử dụng Langchain để băm nhỏ các file Markdown dựa trên Thẻ tiêu đề (Header).
    """
    # 1. Khai báo quy tắc chẻ văn bản theo Header của Markdown
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # 2. Khai báo quy tắc dự phòng: Chẻ theo ký tự
    # chunk_size=1000 ký tự (~150 từ), overlap=100 để các đoạn con giữ context.
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    md_files = glob.glob(os.path.join(input_dir, '*.md'))
    if not md_files:
        logging.warning(f"Không tìm thấy file .md nào trong {input_dir}")
        return

    logging.info(f"Phát hiện tổng cộng {len(md_files)} file Markdown. Khởi động Langchain Chunker...")

    for md_path in md_files:
        file_name = os.path.basename(md_path)
        base_name = os.path.splitext(file_name)[0]
        output_json_path = os.path.join(output_dir, f"{base_name}_chunks.json")
        
        if os.path.exists(output_json_path):
            logging.info(f"File chunks đã tồn tại, bỏ qua: {file_name}")
            continue

        logging.info(f"Đang chunk bài báo: {file_name}...")
        
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                markdown_text = f.read()

            # Bước A: Chunk theo Logic của báo khoa học (Abstract, Introduction, Conclusion,...)
            md_header_splits = markdown_splitter.split_text(markdown_text)
            
            # Bước B: Ép nhỏ lại các đoạn bị dài quá 1000 ký tự
            final_chunks = char_splitter.split_documents(md_header_splits)
            
            # Bước C: Đóng gói toàn bộ văn bản vừa chunk sang định dạng dễ đọc (List các Dictionary)
            processed_chunks = []
            for i, chunk in enumerate(final_chunks):
                processed_chunks.append({
                    "chunk_id": f"{base_name}_{i}",
                    "paper_id": base_name,
                    "metadata": chunk.metadata, # Lưu lại thông tin nó thuộc Header nào (VD: Introduction)
                    "content": chunk.page_content,
                    "char_length": len(chunk.page_content)
                })

            # Xuất ra file JSON
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(processed_chunks, f, ensure_ascii=False, indent=2)

            logging.info(f"Đã tạo ra {len(processed_chunks)} khối văn bản (chunks) nhỏ. Lưu tại: {output_json_path}")

        except Exception as e:
            logging.error(f"Lỗi khi chunking file {file_name}: {e}")

if __name__ == "__main__":
    test_base_dir = '../test_data' if not os.path.exists('test_data') else 'test_data'
    
    input_md_dir = os.path.join(test_base_dir, 'processed', 'markdown')
    output_chunk_dir = setup_chunk_directories(test_base_dir)
    
    logging.info("\n=========================================\nBẮT ĐẦU LAYER CHUNKING (MARKDOWN -> CHUNKS)\n=========================================")
    chunk_markdown_files(input_dir=input_md_dir, output_dir=output_chunk_dir)
    logging.info("Hoàn tất Chunking!")

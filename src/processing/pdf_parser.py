import os

# Fix lỗi [WinError 1314] trên Windows khi HuggingFace tải Model AI về máy
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import glob
import logging
from typing import List

# Import Docling
from docling.document_converter import DocumentConverter

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_processed_directories(base_dir: str = 'test_data'):
    """Tạo thư mục cho dữ liệu đã được xử lý (Processed Data)"""
    processed_dir = os.path.join(base_dir, 'processed', 'markdown')
    os.makedirs(processed_dir, exist_ok=True)
    logging.info(f"Đã kiểm tra cấu trúc thư mục đầu ra: {processed_dir}")
    return processed_dir

def parse_pdfs_to_markdown(input_dir: str, output_dir: str, converter=None):
    """
    Quét toàn bộ file PDF trong thư mục và nhả ra file Markdown.
    Converter được truyền vào hoặc lazy-init 1 lần duy nhất.
    """
    # Tìm kiếm tất cả file PDF
    pdf_files = glob.glob(os.path.join(input_dir, '*.pdf'))
    
    if not pdf_files:
        logging.warning(f"Không tìm thấy file PDF nào trong: {input_dir}")
        return
        
    logging.info(f"Phát hiện tổng cộng {len(pdf_files)} file PDF. Khởi động AI Docling Core...")
    
    # Lazy init: chỉ load model 1 lần
    if converter is None:
        converter = DocumentConverter()
    
    # Duyệt và chuyển đổi từng file một
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        base_name = os.path.splitext(file_name)[0]
        output_md_path = os.path.join(output_dir, f"{base_name}.md")
        
        # Idempotent
        if os.path.exists(output_md_path):
            logging.info(f"File markdown đã tồn tại, bỏ qua parse: {file_name}")
            continue
            
        logging.info(f"Đang phân tích cấu trúc & Bóc tách: {file_name}...")
        try:
            # convert() sẽ tự động nhận diện đoạn văn, 2 cột dọc, title, table...
            result = converter.convert(pdf_path)
            
            # Xuất kết quả ra Markdown
            markdown_content = result.document.export_to_markdown()
            
            # Copy Text ra file
            with open(output_md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
                
            logging.info(f"Bóc tách thành công: Lưu tại {output_md_path}")
            
        except Exception as e:
            logging.error(f"Lỗi khi bóc tách file {file_name}: {e}")
            
if __name__ == "__main__":
    # Chạy mô phỏng thư mục test_data
    test_base_dir = '../test_data' if not os.path.exists('test_data') else 'test_data'
    
    raw_pdf_dir = os.path.join(test_base_dir, 'raw', 'pdf')
    processed_md_dir = setup_processed_directories(test_base_dir)
    
    logging.info("\n=========================================\nBẮT ĐẦU LAYER PROCESSING (PDF -> MARKDOWN)\n=========================================")
    parse_pdfs_to_markdown(input_dir=raw_pdf_dir, output_dir=processed_md_dir)
    logging.info("Hoàn tất Parsing!")

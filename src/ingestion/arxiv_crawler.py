import arxiv
import os
import json
import logging
import ssl
import time
from typing import List, Dict

# Bypass SSL Verification cho urllib
ssl._create_default_https_context = ssl._create_unverified_context

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_directories(base_dir: str = 'data'):
    """Tạo cấu trúc thư mục lưu trữ"""
    dirs = [
        os.path.join(base_dir, 'raw', 'pdf'),
        os.path.join(base_dir, 'raw', 'metadata')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logging.info(f"Đã tạo/kiểm tra thư mục: {d}")
    return dirs

def crawl_arxiv_papers(query: str, max_results: int = 10, download_dir: str = 'data/raw/pdf', metadata_dir: str = 'data/raw/metadata'):
    """
    Crawl bài báo từ ArXiv và tải file PDF.
    
    Args:
        query (str): Câu lệnh truy vấn (VD: 'cat:cs.AI' cho AI, 'LLM' cho từ khóa LLM)
        max_results (int): Số lượng bài báo muốn tải.
        download_dir (str): Thư mục lưu PDF.
        metadata_dir (str): Thư mục lưu Metadata.
    """
    logging.info(f"Bắt đầu tìm kiếm với query: '{query}', max_results: {max_results}")
    
    # Khởi tạo client với độ trễ 5 giây và số lần tự động thử lại (Retry) để chống quá tải
    client = arxiv.Client(page_size=max_results, delay_seconds=5, num_retries=5)
    
    # Định nghĩa phương thức tìm kiếm
    search = arxiv.Search(
        query = query,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.SubmittedDate, # Lấy các bài mới nhất
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers_metadata: List[Dict] = []
    
    # Duyệt qua các kết quả tìm kiếm được
    for r in client.results(search):
        paper_id = r.get_short_id()
        logging.info(f"Đang xử lý bài báo: [{paper_id}] {r.title}")
        
        # 1. Trích xuất Metadata
        metadata = {
            "id": paper_id,
            "title": r.title,
            "authors": [author.name for author in r.authors],
            "summary": r.summary,
            "published": r.published.isoformat(),
            "primary_category": r.primary_category,
            "categories": r.categories,
            "pdf_url": r.pdf_url
        }
        papers_metadata.append(metadata)
        
        # 2. Tải file PDF
        pdf_path = os.path.join(download_dir, f"{paper_id}.pdf")
        if not os.path.exists(pdf_path):
            logging.info(f"Đang tải PDF: {pdf_path}")
            r.download_pdf(dirpath=download_dir, filename=f"{paper_id}.pdf")
        else:
            logging.info(f"File PDF đã tồn tại, bỏ qua tải: {pdf_path}")
            
    # Xóa ký tự gạch chéo/nháy trong query để làm tên file hợp lệ
    safe_query_name = query.replace(':', '_').replace(' ', '_')
    metadata_path = os.path.join(metadata_dir, f"metadata_{safe_query_name}_{max_results}.json")
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(papers_metadata, f, ensure_ascii=False, indent=2)
        
    logging.info(f"Đã lưu metadata của {len(papers_metadata)} bài báo tại {metadata_path}")
    logging.info("Hoàn thành Ingestion!")
    
if __name__ == "__main__":
    # Test_Data
    test_base_dir = 'test_data'
    
    # Bước 1: Khởi tạo/kiểm tra cấu trúc thư mục test
    setup_directories(base_dir=test_base_dir)
    
    # Danh sách cấu hình các chủ đề theo yêu cầu đề bài
    topic_queries = [
        "cat:cs.AI",                    # 1. Trí tuệ nhân tạo (AI)
        "cat:q-fin.GN",                 # 2. Tài chính (Quantitative Finance)
        "cat:eess.SY",                  # 3. Cơ khí/Điện tử/Hệ thống (Systems and Control)
        "all:military OR all:defense"   # 4. Quân sự / Quốc phòng (Keyword search)
    ]
    
    # Quét qua từng chủ đề vòng lặp
    for query in topic_queries:
        logging.info(f"\n=========================================\nBẮT ĐẦU CÀO CHỦ ĐỀ: {query}\n=========================================")
        crawl_arxiv_papers(
            query=query, 
            max_results=2,  # Tạm thời để 2 bài/chủ đề để test
            download_dir=f"{test_base_dir}/raw/pdf",
            metadata_dir=f"{test_base_dir}/raw/metadata"
        )
        # Lưu ý: Bắt buộc phải có khoảng nghỉ giữa nhiều request lớn để không bị block IP
        logging.info("Nghỉ 5 giây để tránh lỗi Rate Limit (HTTP 429) trước khi lấy mảng tiếp theo...")
        time.sleep(5)

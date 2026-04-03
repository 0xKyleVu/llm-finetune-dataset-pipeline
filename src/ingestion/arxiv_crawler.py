import arxiv
import os
import json
import logging
import ssl
import time
from typing import List, Dict

# SSL: Ưu tiên dùng certifi nếu có, chỉ fallback unverified khi không cài được
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
except ImportError:
    ssl._create_default_https_context = ssl._create_unverified_context
    logging.warning("certifi not found — SSL verification disabled. Run: pip install certifi")

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
        
        # 2. Lưu metadata theo từng paper (1 file/paper → dễ JOIN downstream)
        per_paper_path = os.path.join(metadata_dir, f"{paper_id}.json")
        if not os.path.exists(per_paper_path):
            with open(per_paper_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 3. Tải file PDF
        pdf_path = os.path.join(download_dir, f"{paper_id}.pdf")
        if not os.path.exists(pdf_path):
            logging.info(f"Đang tải PDF: {pdf_path}")
            r.download_pdf(dirpath=download_dir, filename=f"{paper_id}.pdf")
        else:
            logging.info(f"File PDF đã tồn tại, bỏ qua tải: {pdf_path}")
            
    logging.info(f"Đã lưu metadata của {len(papers_metadata)} bài báo.")
    logging.info("Hoàn thành Ingestion!")
    
if __name__ == "__main__":
    # Test_Data
    test_base_dir = 'test_data'
    
    # Khởi tạo/kiểm tra cấu trúc thư mục test
    setup_directories(base_dir=test_base_dir)
    
    # Danh sách cấu hình các chủ đề
    topic_queries = [
        "cat:cs.AI", "cat:cs.LG",           # 1. Trí tuệ nhân tạo & Machine Learning
        "cat:cs.CR",                        # 2. Ngành Bảo mật & Mật mã học
        "cat:cs.IT", "cat:cs.NI",           # 3. Mạng Internet & Lý thuyết thông tin
        "cat:eess.SY", "cat:eess.SP",       # 4. Điện tử & Kỹ thuật hệ thống
        "cat:q-fin.GN", "cat:q-fin.CP",     # 5. Tài chính định lượng
        "all:military OR all:defense"       # 6. Quân sự / Quốc phòng (Giữ OR vì đây là tìm kiếm từ khóa đồng nghĩa)
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
        # Lưu ý: Khoảng nghỉ 5s giữa nhiều request lớn để không bị block IP
        logging.info("Nghỉ 5 giây để tránh lỗi Rate Limit (HTTP 429) trước khi lấy mảng tiếp theo...")
        time.sleep(5)

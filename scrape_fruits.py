import os
from icrawler.builtin import BingImageCrawler
from pathlib import Path

# Thư mục chứa dataset của bạn
DATASET_DIR = r"D:\CPPHM\Final_Fruit_Dataset_500Plus"
TARGET_COUNT = 500

def scrape_missing_images():
    print(f"Bắt đầu quá trình quét và cào thêm ảnh tại: {DATASET_DIR}")
    
    if not os.path.exists(DATASET_DIR):
        print(f"Thư mục {DATASET_DIR} không tồn tại!")
        return

    # Lặp qua tất cả các lớp trái cây trong thư mục
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    for fruit_class in classes:
        class_dir = os.path.join(DATASET_DIR, fruit_class)
        existing_images = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
        
        print(f"\n--- Lớp: {fruit_class} ---")
        print(f"Số lượng ảnh hiện tại: {existing_images}")
        
        if existing_images < TARGET_COUNT:
            need_to_download = TARGET_COUNT - existing_images
            print(f"Còn thiếu {need_to_download} ảnh. Bắt đầu cào thêm từ Bing...")
            
            # Cấu hình crawler
            bing_crawler = BingImageCrawler(
                downloader_threads=4,
                storage={'root_dir': class_dir}
            )
            
            # Tiến hành cào ảnh 
            keyword = f"{fruit_class} fruit"
            bing_crawler.crawl(
                keyword=keyword, 
                max_num=need_to_download,
                # Tùy chọn lọc: chỉ tải hình ảnh minh họa thực tế
                filters={'type': 'photo'}
            )
            print(f"Đã cào xong bù thêm cho {fruit_class}.")
        else:
            print(f"> Đã đạt (và vượt) mốc {TARGET_COUNT}+ ảnh. Bỏ qua.")

if __name__ == '__main__':
    scrape_missing_images()
    print("\nHOÀN THÀNH TOÀN BỘ QUÁ TRÌNH!")

from ultralytics import YOLO

def train():
    print("=== BẮT ĐẦU TREAN MÔ HÌNH NHẬN DIỆN TRÁI CÂY (5 LỚP) ===")
    
    # Sử dụng YOLOv8 phiên bản Nano (nhẹ nhất, nhanh nhất) để huấn luyện
    model = YOLO("yolov8n.pt")
    
    # Chỉ định đường dẫn tới file YAML khai báo Dataset (Nằm trong ổ D)
    yaml_path = r"D:\CPPHM\YOLO_Demo_Dataset\demo.yaml"
    
    # Bắt đầu luyện với 10 vòng lặp (Epochs). Tăng epochs nếu muốn độ chuẩn xác cao hơn
    results = model.train(
        data=yaml_path,
        epochs=10,        # 10 vòng lặp 
        imgsz=416,        # Kích thước ảnh đầu vào (nhỏ cho nhanh)
        project=r"D:\CPPHM\runs", # Nơi lưu kết quả trong ổ D
        name="fruit_demo", # Tên folder chứa file weights
        batch=16,
        device="cpu",     # Chạy trên CPU để tương thích mọi máy
        plots=True        # Vẽ biểu đồ Accuracy lúc sau để xem
    )
    
    print("=== HOÀN TẤT HUẤN LUYỆN! ===")
    print("Mô hình được lưu tại: D:\\CPPHM\\runs\\fruit_demo\\weights\\best.pt")

if __name__ == "__main__":
    train()

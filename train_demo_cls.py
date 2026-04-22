from ultralytics import YOLO

def train_classification():
    print("=== BẮT ĐẦU TREAN MÔ HÌNH PHÂN LOẠI (CLASSIFICATION) ===")
    
    # Sử dụng mô hình phân loại (Classification) thay vì Object Detection
    model = YOLO("yolov8n-cls.pt")
    
    # Đường dẫn thẳng tới thư mục gốc chứa 25 lớp
    dataset_dir = r"D:\CPPHM\YOLO_Demo_Cls_25"
    
    # Bắt đầu luyện
    results = model.train(
        data=dataset_dir,
        epochs=20,        # Nâng lên 20 vòng lặp để AI phân tích thật kỹ
        patience=5,       # Sẽ dừng sớm nếu 5 vòng liên tiếp không tụt thêm lỗi (Chống lãng phí time)
        imgsz=224,        # Classification thì chỉ cần 224x224 là đủ, nhẹ cực kỳ
        project=r"D:\CPPHM\runs_cls", 
        name="fruit_demo_cls", 
        device="cpu",     
        plots=True        
    )
    
    print("=== HOÀN TẤT HUẤN LUYỆN! ===")
    print("Mô hình được lưu tại: D:\\CPPHM\\runs_cls\\fruit_demo_cls\\weights\\best.pt")

if __name__ == "__main__":
    train_classification()

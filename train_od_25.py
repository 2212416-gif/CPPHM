from ultralytics import YOLO

def train_od_25():
    print("=== BAT DAU TRAIN MO HINH OBJECT DETECTION 25 LOP TRAI CAY ===")
    
    # Dùng YOLOv8 Nano Object Detection (Không phải -cls)
    model = YOLO("yolov8n.pt")
    
    yaml_path = r"D:\CPPHM\YOLO_OD_25_Dataset\od25.yaml"
    
    results = model.train(
        data=yaml_path,
        epochs=20,           # 20 vòng lặp để AI học kỹ
        patience=5,          # Tự dừng sớm nếu không còn cải thiện
        imgsz=416,           # Kích thước ảnh đầu vào
        batch=8,             # Giảm xuống 8 để không bị tràn RAM
        project=r"D:\CPPHM\runs_od25",
        name="fruit_od_25",
        device="cpu",
        plots=True
    )
    
    print("=== HOAN TAT HUAN LUYEN! ===")
    print(r"Mo hinh duoc luu tai: D:\CPPHM\runs_od25\fruit_od_25\weights\best.pt")

if __name__ == "__main__":
    train_od_25()

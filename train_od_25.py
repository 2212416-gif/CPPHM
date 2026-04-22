from ultralytics import YOLO

def train_od_25_pro():
    print("=== BAT DAU HUAN LUYEN AI SIEU CAP (OBJECT DETECTION 25 LOP) ===")
    
    # Su dung YOLOv8 Small (v8s) - Thong minh hon ban Nano nhieu
    model = YOLO("yolov8s.pt")
    
    yaml_path = r"D:\CPPHM\YOLO_OD_25_Dataset\od25.yaml"
    
    results = model.train(
        data=yaml_path,
        epochs=30,           # Tang len 30 vong de AI nhin nhan tinh vi hon
        patience=10,         # Kien nhan hon truoc khi dung
        imgsz=640,           # Tang do phan giai len 640 de nhin ro chi tiet
        batch=4,             # Giam batch de khong bi treo may do mo hinh lon hon
        project=r"D:\CPPHM\runs_od25",
        name="fruit_od_25_pro",
        device="cpu",        # Chay tren CPU
        augment=True,        # Bat tinh nang tu bien hoa du lieu (xoay, lat, nhieu)
        plots=True
    )
    
    print("=== HOAN TAT HUAN LUYEN SIEU CAP! ===")
    print(r"Mo hinh xịn nhat tai: D:\CPPHM\runs_od25\fruit_od_25_pro\weights\best.pt")

if __name__ == "__main__":
    train_od_25_pro()

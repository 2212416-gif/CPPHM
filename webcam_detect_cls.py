import cv2
import os
import numpy as np
from ultralytics import YOLO

def detect_from_webcam_cls():
    # Tìm tự động thư mục huấn luyện mới nhất (tránh lỗi đổi tên sinh ra khi train nhiều lần)
    base_run_dir = r"D:\CPPHM\runs_cls"
    model_path = r"D:\CPPHM\runs_cls\fruit_demo_cls\weights\best.pt" # Default
    
    if os.path.exists(base_run_dir):
        # Liệt kê tất cả thư mục bắt đầu bằng fruit_demo_cls
        subdirs = [os.path.join(base_run_dir, d) for d in os.listdir(base_run_dir) if d.startswith("fruit_demo_cls")]
        if subdirs:
            # Lấy thư mục mới được tạo gần đây nhất
            latest_dir = max(subdirs, key=os.path.getmtime)
            model_path = os.path.join(latest_dir, "weights", "best.pt")

    if not os.path.exists(model_path):
        print(f"Không tìm thấy mô hình tại {model_path}!")
        print("Vui lòng chạy file `train_demo_cls.py` trước.")
        return

    print("=== ĐANG TẢI MÔ HÌNH PHÂN LOẠI LÊN BỘ NHỚ ===")
    model = YOLO(model_path)
    print("=== TẢI XONG! ĐANG MỞ WEBCAM... (Bấm 'q' để thoát) ===")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Lỗi: Không thể kết nối với Webcam. Vui lòng kiểm tra lại Camera!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # YOLOv8 Classification không vẽ box, nó chỉ trả về xác suất của lớp cao nhất
        results = model(frame, verbose=False)
        
        # Tiết xuất kết quả cao nhất
        top1_index = results[0].probs.top1
        confidence = results[0].probs.top1conf.item()
        predicted_class = results[0].names[top1_index]
        
        # Tạo bảng chữ hiển thị kết quả trên góc trái màn hình
        # Chỉ hiển thị kết quả nếu độ tự tin > 40% (xem như bớt độ trôi màu)
        text = f"{predicted_class}: {confidence*100:.1f}%" if confidence > 0.4 else "Khong ro..."
        color = (0, 255, 0) if confidence > 0.4 else (0, 0, 255)
        
        # Vẽ một thanh nền đen (hoặc xanh) để dễ đọc chữ
        cv2.rectangle(frame, (10, 10), (500, 60), (0,0,0), -1)
        cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        cv2.imshow("Nhan Vien Trai Cay - Image Classification", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_from_webcam_cls()

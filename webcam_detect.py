import cv2
import os
from ultralytics import YOLO

def detect_from_webcam():
    # Đường dẫn file AI mô hình (Sau khi Training xong)
    model_path = r"D:\CPPHM\runs\fruit_demo\weights\best.pt"
    
    if not os.path.exists(model_path):
        print(f"Không tìm thấy mô hình tại {model_path}!")
        print("Vui lòng chạy file `train_demo.py` để huấn luyện ra mô hình trước.")
        return

    print("=== ĐANG TẢI MÔ HÌNH LÊN BỘ NHỚ ===")
    model = YOLO(model_path)
    print("=== TẢI XONG! ĐANG MỞ WEBCAM... (Bấm 'q' để thoát) ===")
    
    # Mở Camera Default (0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Lỗi: Không thể kết nối với Webcam. Vui lòng kiểm tra lại Camera!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi đọc dữ liệu từ Webcam.")
            break
            
        # YOLOv8 có hàm xử lý và vẽ luôn Bounding Box lên Frame tự động
        results = model(frame, conf=0.5, verbose=False) # Lọc với Confidence >= 50%
        
        # results[0].plot() trả về 1 ma trận Frame chứa hình ảnh đã vẽ
        annotated_frame = results[0].plot()
        
        # Hiện hình ảnh lên màn hình
        cv2.imshow("Nhan Vien Trai Cay - OpenCV YOLOv8", annotated_frame)
        
        # Bấm phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_from_webcam()

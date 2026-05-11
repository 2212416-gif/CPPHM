import cv2
import os
from ultralytics import YOLO

def detect_od_webcam():
    # 1. Tìm mô hình mới nhất tự động (Bao gồm cả bản PRO nếu đã xong)
    base_dir = r"D:\CPPHM\runs_od25"
    model_path = ""
    
    if os.path.exists(base_dir):
        # Lấy tất cả file best.pt trong các thư mục con của runs_od25
        all_weights = []
        for root, dirs, files in os.walk(base_dir):
            if "best.pt" in files:
                path = os.path.join(root, "best.pt")
                all_weights.append((path, os.path.getmtime(path)))
        
        if all_weights:
            # Sắp xếp theo thời gian để lấy file mới nhất
            all_weights.sort(key=lambda x: x[1], reverse=True)
            model_path = all_weights[0][0]

    if not model_path or not os.path.exists(model_path):
        print(f"Chua tim thay mo hinh tai {base_dir}")
        print("Vui long doi AI train xong hoac kiem tra lai thu muc.")
        return

    print(f"=== DANG DUNG MO HINH: {model_path} ===")
    model = YOLO(model_path)
    
    # Lay danh sach ten tieng Viet tu file model (neu co)
    class_names = model.names 

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Loi: Khong mo duoc Webcam!")
        return

    # Ngưỡng tự tin (Tăng lên 0.5 để chống nhảy nhót)
    CONF_THRESHOLD = 0.5

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Chạy nhận diện
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        
        # Vẽ thủ công để tùy chỉnh % và chữ
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Lấy tọa độ
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Lấy độ tự tin và lớp
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = class_names[cls]
                
                # Chuyển đổi sang % (Ví dụ: 0.85 -> 85%)
                percent = int(conf * 100)
                label = f"{name} ({percent}%)"
                
                # Vẽ khung hình (Màu xanh neon cho chuyên nghiệp)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Vẽ nền cho nhãn chữ
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                
                # Ghi chữ nhãn
                cv2.putText(frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow("DEMO NHAN DIEN TRAI CAY 25 LOP", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_od_webcam()

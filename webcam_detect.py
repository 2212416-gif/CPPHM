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

    # Dictionary để lưu thông tin cố định: {track_id: (label_name, confidence)}
    fixed_info = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi đọc dữ liệu từ Webcam.")
            break
            
        # Sử dụng model.track để cấp ID duy nhất cho từng trái cây
        results = model.track(frame, persist=True, conf=0.5, verbose=False)
        
        # Kiểm tra nếu có vật thể được nhận diện và có ID
        if results[0].boxes.id is not None:
            # Lặp qua từng vật thể tìm được
            for box_data in results[0].boxes:
                if box_data.id is None: continue
                
                track_id = int(box_data.id)
                cls_id = int(box_data.cls)
                conf_val = float(box_data.conf)
                box = box_data.xyxy[0].int().tolist() # [x1, y1, x2, y2]
                
                # Nếu ID này lần đầu xuất hiện, lưu thông tin lại
                if track_id not in fixed_info:
                    label_name = model.names[cls_id]
                    fixed_info[track_id] = (label_name, conf_val * 100)
                
                # Lấy thông tin cố định từ "kho"
                name, fixed_conf = fixed_info[track_id]
                
                # Vẽ thủ công để đảm bảo hiển thị đúng thông tin đã chốt
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label_text = f"ID:{track_id} {name} {fixed_conf:.1f}%"
                cv2.putText(frame, label_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Hiện hình ảnh lên màn hình
        cv2.imshow("Nhan Dien Trai Cay (Fixed) - YOLOv8", frame)
        
        # Bấm phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_from_webcam()

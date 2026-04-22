import cv2
import os
from ultralytics import YOLO

def detect_od_webcam():
    # Tìm mô hình OD mới nhất tự động
    base_dir = r"D:\CPPHM\runs_od25"
    model_path = r"D:\CPPHM\runs_od25\fruit_od_25\weights\best.pt"

    if os.path.exists(base_dir):
        subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
                   if d.startswith("fruit_od_25")]
        if subdirs:
            latest_dir = max(subdirs, key=os.path.getmtime)
            candidate = os.path.join(latest_dir, "weights", "best.pt")
            if os.path.exists(candidate):
                model_path = candidate

    if not os.path.exists(model_path):
        print(f"Khong tim thay mo hinh tai: {model_path}")
        print("Vui long chay train_od_25.py de huan luyen mo hinh truoc!")
        return

    print(f"=== DANG TAI MO HINH OD TU: {model_path} ===")
    model = YOLO(model_path)
    print("=== TAI XONG! MO WEBCAM... (Bam 'q' de thoat) ===")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Loi: Khong the ket noi Webcam!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chạy Object Detection - tự vẽ bbox + tên lớp lên frame
        results = model(frame, conf=0.35, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("NHAN DIEN TRAI CAY - Object Detection 25 Lop", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_od_webcam()

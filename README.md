# Dự án Nhận diện Trái cây - YOLOv8 Object Detection

Dự án này sử dụng YOLOv8 để nhận diện 25 loại trái cây khác nhau thông qua Webcam.

## Các tính năng chính:
- Tự động cào dữ liệu ảnh từ Bing.
- Chuẩn bị Dataset Object Detection (Bounding Box tự động).
- Huấn luyện mô hình YOLOv8 (bản Pro - v8s).
- Nhận diện thời gian thực qua Webcam với nhãn Tiếng Việt.

## Danh sách 25 loại trái cây:
Táo, Chuối, Cà chua, Cam, Chanh, Bơ, Việt Quất, Dưa Vàng, Khế, Anh Đào, Dừa, Bắp, Cà Tím, Nho, Ổi, Kiwi, Vải, Xoài, Đào, Lê, Ớt Chuông, Thơm, Lựu, Dâu Tây, Dưa Hấu.

## Hướng dẫn sử dụng:
1. Cài đặt thư viện: `pip install ultralytics opencv-python`
2. Chạy Webcam: `python webcam_od_25.py`
3. Nếu muốn huấn luyện lại:
   - `python prepare_od_25_classes.py`
   - `python train_od_25.py`

## Cấu trúc thư mục quan trọng:
- `runs_od25/`: Chứa file model đã huấn luyện (`best.pt`).
- các file `prepare_*.py`: Script xử lý dữ liệu.
- `webcam_od_25.py`: Script chạy chính.

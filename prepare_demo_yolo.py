import os
import shutil
import random
from PIL import Image

# Cấu hình đường dẫn (tối ưu vào ổ D)
SOURCE_DIR = r"D:\CPPHM\Final_Fruit_Dataset_500Plus"
DEST_DIR = r"D:\CPPHM\YOLO_Demo_Dataset"

# 5 lớp trái cây Demo
TARGET_CLASSES = ["Apple Braeburn", "Banana", "Tomato", "Orange", "Lemon"]

def get_foreground_bbox(img):
    """
    Tự động tính khung vuông (Bounding Box) bằng cách loại bỏ viền trắng/viền sáng.
    Vì dữ liệu dạng Fruits-360 thường có nền trắng xung quanh.
    """
    img_gray = img.convert("L")
    # Lấy ma trận bbox của những vùng không phải màu trắng
    # Inverter: ImageOps.invert(img_gray)
    bbox = img_gray.point(lambda p: p < 250 and 255).getbbox()
    
    if bbox:
        left, upper, right, lower = bbox
        width, height = img.size
        # Chuẩn hoá tỷ lệ 0..1 cho YOLO
        x_center = ((left + right) / 2.0) / width
        y_center = ((upper + lower) / 2.0) / height
        bw = (right - left) / width
        bh = (lower - upper) / height
        
        # Thêm 1 chút padding nhỏ để bao trọn
        bw = min(1.0, bw * 1.05)
        bh = min(1.0, bh * 1.05)
        
        return x_center, y_center, bw, bh
    else:
        # Nếu không bắt được viền ảnh -> Lấy 90% diện tích tâm ảnh
        return 0.5, 0.5, 0.9, 0.9

def prepare_yolo_dataset():
    if not os.path.exists(SOURCE_DIR):
        print(f"Lỗi: Không tìm thấy {SOURCE_DIR}")
        return

    # Tạo cấu trúc thư mục YOLO
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(DEST_DIR, folder), exist_ok=True)

    print("=== BẮT ĐẦU CHUYỂN ĐỔI SANG YOLO DATASET (Gắn nhãn O.D) ===")

    for class_idx, class_name in enumerate(TARGET_CLASSES):
        class_dir = os.path.join(SOURCE_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Bo qua: {class_name} vi khong ton tai")
            continue
            
        all_images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Giới hạn lấy tối đa 500 tấm cho nhẹ, hoặc lấy hết
        random.shuffle(all_images)
        all_images = all_images[:600] 
        print(f"Xu ly lop {class_name} (ID: {class_idx}) voi {len(all_images)} anh...")

        # Chia 80% train, 20% validation
        split_idx = int(0.8 * len(all_images))
        train_imgs = all_images[:split_idx]
        val_imgs = all_images[split_idx:]

        for split_name, img_subset in zip(["train", "val"], [train_imgs, val_imgs]):
            for img_file in img_subset:
                src_path = os.path.join(class_dir, img_file)
                # Đổi tên file để tránh trùng lặp giữa các class
                new_basename = f"{class_name.replace(' ', '_')}_{img_file}"
                
                # Coppy ảnh
                dst_img_path = os.path.join(DEST_DIR, 'images', split_name, new_basename)
                shutil.copy2(src_path, dst_img_path)

                # Mở ảnh để tính Bounding Box tự động
                try:
                    with Image.open(src_path) as img:
                        xc, yc, w, h = get_foreground_bbox(img)
                except Exception as e:
                    xc, yc, w, h = 0.5, 0.5, 0.9, 0.9 # Default fallback

                # Ghi Bbox format YOLO `.txt`
                txt_filename = new_basename.rsplit('.', 1)[0] + '.txt'
                dst_lbl_path = os.path.join(DEST_DIR, 'labels', split_name, txt_filename)
                
                with open(dst_lbl_path, 'w', encoding='utf-8') as f:
                    # Cấu trúc: class_id x_center y_center width height
                    f.write(f"{class_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # Sinh file demo.yaml cấu hình dataset
    yaml_path = os.path.join(DEST_DIR, "demo.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {DEST_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(TARGET_CLASSES)}\n")
        f.write("names:\n")
        for idx, cls_name in enumerate(TARGET_CLASSES):
            f.write(f"  {idx}: {cls_name}\n")
            
    print("=== ĐÃ TẠO DATASET TỰ ĐỘNG THÀNH CÔNG TẠI D:\\CPPHM\\YOLO_Demo_Dataset ===")

if __name__ == "__main__":
    prepare_yolo_dataset()

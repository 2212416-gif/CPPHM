import os
import shutil
import random
import cv2
import numpy as np

# Cấu hình đường dẫn
SOURCE_DIR = r"D:\CPPHM\Final_Fruit_Dataset_500Plus"
DEST_DIR   = r"D:\CPPHM\YOLO_OD_25_Dataset"

TARGET_CLASSES = [
    "Apple Braeburn", "Banana", "Tomato", "Orange", "Lemon",
    "Avocado", "Blueberry", "Cantaloupe", "Carambula", "Cherry",
    "Cocos", "Corn", "Eggplant", "Grape Blue", "Guava",
    "Kiwi", "Lychee", "Mango", "Peach", "Pear",
    "Pepper Red", "Pineapple", "Pomegranate", "Strawberry", "Watermelon"
]

DISPLAY_NAMES = [
    "Tao", "Chuoi", "Ca chua", "Cam", "Chanh",
    "Qua Bo", "Viet Quat", "Dua Vang", "Khe", "Anh Dao",
    "Dua", "Bap Ngo", "Ca Tim", "Nho Tim", "Oi",
    "Kiwi", "Vai", "Xoai", "Dao", "Le",
    "Ot Chuong Do", "Thom Dua", "Luu", "Dau Tay", "Dua Hau"
]

def get_accurate_bbox_cv2(img_path):
    """Sử dụng OpenCV để tìm khung hình cực chuẩn cho trái cây."""
    img = cv2.imread(img_path)
    if img is None:
        return 0.5, 0.5, 0.8, 0.8
    
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ngưỡng: Lấy những vùng tối hơn màu trắng nền (thường nền trắng là 255)
    # Ta lấy ngưỡng 250 để loại bỏ bóng đổ nhẹ
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Tìm các điểm ảnh không phải nền trắng
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        height, width = gray.shape
        
        # Tính toán format YOLO (0..1)
        xc = (x + w/2) / width
        yc = (y + h/2) / height
        bw = w / width
        bh = h / height
        
        # Thêm 2% padding cho đẹp
        bw = min(1.0, bw * 1.02)
        bh = min(1.0, bh * 1.02)
        return xc, yc, bw, bh
    
    return 0.5, 0.5, 0.8, 0.8

def prepare_od_dataset_v2():
    if not os.path.exists(SOURCE_DIR):
        print(f"Loi: Khong tim thay {SOURCE_DIR}")
        return

    # Xoá cũ tạo mới để đảm bảo sạch nhãn
    if os.path.exists(DEST_DIR):
        print("Dang lam sach thu muc cu...")
        shutil.rmtree(DEST_DIR)

    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(DEST_DIR, folder), exist_ok=True)

    print("=== BAT DAU TAO DATASET OD CHUAN XAC CAO (25 LOP) ===")

    for class_idx, class_name in enumerate(TARGET_CLASSES):
        class_dir = os.path.join(SOURCE_DIR, class_name)
        if not os.path.exists(class_dir):
            continue

        all_images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(all_images)
        all_images = all_images[:500] 
        print(f"  Dạng xử lý: {class_name}...")

        split_idx = int(0.8 * len(all_images))
        for i, img_file in enumerate(all_images):
            split_name = "train" if i < split_idx else "val"
            src_path = os.path.join(class_dir, img_file)
            new_name = f"{class_name.replace(' ', '_')}_{img_file}"
            
            # Copy ảnh
            dst_img = os.path.join(DEST_DIR, 'images', split_name, new_name)
            shutil.copy2(src_path, dst_img)

            # Tính Bbox chuẩn
            xc, yc, w, h = get_accurate_bbox_cv2(src_path)

            # Ghi nhãn
            txt_name = new_name.rsplit('.', 1)[0] + '.txt'
            dst_lbl  = os.path.join(DEST_DIR, 'labels', split_name, txt_name)
            with open(dst_lbl, 'w') as f:
                f.write(f"{class_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # File YAML
    yaml_path = os.path.join(DEST_DIR, "od25.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        dest_path = DEST_DIR.replace('\\', '/')
        f.write(f"path: {dest_path}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(TARGET_CLASSES)}\n")
        f.write("names:\n")
        for idx, display in enumerate(DISPLAY_NAMES):
            f.write(f"  {idx}: {display}\n")

    print(f"\n=== HOAN TAT! NHAN DA DUOC SUA CHUAN TAI: {DEST_DIR} ===")

if __name__ == "__main__":
    prepare_od_dataset_v2()

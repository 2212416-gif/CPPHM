import os
import shutil
import random
from PIL import Image

# Cấu hình đường dẫn
SOURCE_DIR = r"D:\CPPHM\Final_Fruit_Dataset_500Plus"
DEST_DIR   = r"D:\CPPHM\YOLO_OD_25_Dataset"

# 25 lớp trái cây với tên rõ ràng cho file YAML
TARGET_CLASSES = [
    "Apple Braeburn", "Banana", "Tomato", "Orange", "Lemon",
    "Avocado", "Blueberry", "Cantaloupe", "Carambula", "Cherry",
    "Cocos", "Corn", "Eggplant", "Grape Blue", "Guava",
    "Kiwi", "Lychee", "Mango", "Peach", "Pear",
    "Pepper Red", "Pineapple", "Pomegranate", "Strawberry", "Watermelon"
]

# Tên hiển thị thân thiện bằng Tiếng Việt (hiện trên màn hình Camera)
DISPLAY_NAMES = [
    "Tao", "Chuoi", "Ca chua", "Cam", "Chanh",
    "Qua Bo", "Viet Quat", "Dua Vang", "Khe", "Anh Dao",
    "Dua", "Bap Ngo", "Ca Tim", "Nho Tim", "Oi",
    "Kiwi", "Vai", "Xoai", "Dao", "Le",
    "Ot Chuong Do", "Thom Dua", "Luu", "Dau Tay", "Dua Hau"
]

def get_foreground_bbox(img):
    """Tự động tính Bounding Box bằng cách loại bỏ nền trắng."""
    img_gray = img.convert("L")
    bbox = img_gray.point(lambda p: p < 250 and 255).getbbox()
    if bbox:
        left, upper, right, lower = bbox
        width, height = img.size
        x_center = ((left + right) / 2.0) / width
        y_center = ((upper + lower) / 2.0) / height
        bw = min(1.0, (right - left) / width * 1.05)
        bh = min(1.0, (lower - upper) / height * 1.05)
        return x_center, y_center, bw, bh
    else:
        return 0.5, 0.5, 0.9, 0.9

def prepare_od_dataset():
    if not os.path.exists(SOURCE_DIR):
        print(f"Loi: Khong tim thay {SOURCE_DIR}")
        return

    # Tạo cấu trúc thư mục YOLO OD
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(DEST_DIR, folder), exist_ok=True)

    print("=== BAT DAU TAO YOLO OD DATASET CHO 25 LOP TRAI CAY ===")

    for class_idx, class_name in enumerate(TARGET_CLASSES):
        class_dir = os.path.join(SOURCE_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"  Canh bao: Khong tim thay [{class_name}], bo qua.")
            continue

        all_images = [f for f in os.listdir(class_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(all_images)
        all_images = all_images[:500]  # Tối đa 500 ảnh/lớp
        print(f"  [{class_idx+1}/25] {class_name}: {len(all_images)} anh")

        split_idx = int(0.8 * len(all_images))
        train_imgs = all_images[:split_idx]
        val_imgs   = all_images[split_idx:]

        for split_name, img_subset in zip(["train", "val"], [train_imgs, val_imgs]):
            for img_file in img_subset:
                src_path = os.path.join(class_dir, img_file)
                safe_name = class_name.replace(' ', '_')
                new_basename = f"{safe_name}_{img_file}"

                # Copy ảnh
                dst_img = os.path.join(DEST_DIR, 'images', split_name, new_basename)
                shutil.copy2(src_path, dst_img)

                # Tính và ghi Bounding Box
                try:
                    with Image.open(src_path) as img:
                        xc, yc, w, h = get_foreground_bbox(img)
                except Exception:
                    xc, yc, w, h = 0.5, 0.5, 0.9, 0.9

                txt_name = new_basename.rsplit('.', 1)[0] + '.txt'
                dst_lbl  = os.path.join(DEST_DIR, 'labels', split_name, txt_name)
                with open(dst_lbl, 'w') as f:
                    f.write(f"{class_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # Tạo file YAML cấu hình
    yaml_path = os.path.join(DEST_DIR, "od25.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        dest_dir_forward = DEST_DIR.replace('\\', '/')
        f.write(f"path: {dest_dir_forward}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(TARGET_CLASSES)}\n")
        f.write("names:\n")
        for idx, display in enumerate(DISPLAY_NAMES):
            f.write(f"  {idx}: {display}\n")

    print("\n=== HOAN TAT! DATASET OD 25 LOP DA DUOC TAO TAI: ===")
    print(f"  {DEST_DIR}")
    print(f"  File YAML: {yaml_path}")

if __name__ == "__main__":
    prepare_od_dataset()

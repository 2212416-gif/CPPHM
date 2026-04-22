import os
import shutil

def prepare_5_classes():
    source_dir = r"D:\CPPHM\Final_Fruit_Dataset_500Plus"
    dest_dir = r"D:\CPPHM\YOLO_Demo_Cls_Dataset"
    
    classes_to_keep = ["Apple Braeburn", "Banana", "Tomato", "Orange", "Lemon"]
    
    print("=== ĐANG TẠO THƯ MỤC CỦA 5 LOẠI TRÁI CÂY (DÀNH CHO PHÂN LOẠI) ===")
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    for cls in classes_to_keep:
        src = os.path.join(source_dir, cls)
        dst = os.path.join(dest_dir, cls)
        
        if os.path.exists(src):
            if not os.path.exists(dst):
                print(f"Copying {cls}...")
                shutil.copytree(src, dst)
            else:
                print(f"Đã có sãn {cls}.")
        else:
            print(f"Cảnh báo: không tìm thấy {cls} trong thư mục gốc.")
            
    print("=== HOÀN TẤT SAO CHÉP ===")

if __name__ == "__main__":
    prepare_5_classes()

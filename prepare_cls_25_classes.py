import os
import shutil

def prepare_25_classes():
    source_dir = r"D:\CPPHM\Final_Fruit_Dataset_500Plus"
    dest_dir = r"D:\CPPHM\YOLO_Demo_Cls_25"
    
    classes_to_keep = [
        "Apple Braeburn", "Banana", "Tomato", "Orange", "Lemon",
        "Avocado", "Blueberry", "Cantaloupe", "Carambula", "Cherry",
        "Cocos", "Corn", "Eggplant", "Grape Blue", "Guava",
        "Kiwi", "Lychee", "Mango", "Peach", "Pear",
        "Pepper Red", "Pineapple", "Pomegranate", "Strawberry", "Watermelon"
    ]
    
    print("=== ĐANG TẠO THƯ MỤC CỦA 25 LOẠI TRÁI CÂY (DÀNH CHO PHÂN LOẠI) ===")
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    for cls in classes_to_keep:
        src = os.path.join(source_dir, cls)
        dst = os.path.join(dest_dir, cls)
        print(f"Xử lý: {cls}...")
        
        if os.path.exists(src):
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
            else:
                pass
        else:
            print(f"Cảnh báo: không tìm thấy {cls} trong thư mục gốc.")
            
    print("=== HOÀN TẤT SAO CHÉP 25 THƯ MỤC! ===")

if __name__ == "__main__":
    prepare_25_classes()

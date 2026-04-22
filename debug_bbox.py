import cv2
import numpy as np

img_path = r"D:\CPPHM\Final_Fruit_Dataset_500Plus\Apple Braeburn\0_100.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
coords = cv2.findNonZero(thresh)
if coords is not None:
    x, y, w, h = cv2.boundingRect(coords)
    print(f"Bbox: x={x}, y={y}, w={w}, h={h}, img_size={gray.shape}")
else:
    print("No non-zero pixels found")

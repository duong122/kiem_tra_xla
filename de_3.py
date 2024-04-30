import cv2 as cv
import numpy as np

I = cv.imread(r"D:\Image_processing\color_split.png")
cv.imshow("origin image", I)
Ihsv = cv.cvtColor(I, cv.COLOR_BGR2HSV)
h, s, v = cv.split(Ihsv)
cv.imshow("kenh mau v", v)
v = np.uint8(v)
max_v = v.max()
print("Muc sang lon nhat cua kenh mau v la: ", max_v)

kernel = np.ones((5, 5), np.float32) / 25
img = cv.filter2D(I, -1, kernel)
cv.imshow("anh sau khi loc trung binh la ", img)



























cv.waitKey(0)
cv.destroyAllWindows()
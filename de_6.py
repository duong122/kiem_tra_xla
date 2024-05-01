import cv2 as cv
import numpy as np

I = cv.imread(r"D:\Image_processing\test_contour.png", cv.IMREAD_COLOR)
cv.imshow("anh mau doc vao la ", I)

Ihsv = cv.cvtColor(I, cv.COLOR_BGR2HSV)
h,s, v = cv.split(Ihsv)
cv.imshow("Kenh mau h cua Ihsv la", h)
max_s = s.max()
print("Gia tri muc sang lon nhat cua kenh s la: ", max_s)


























cv.waitKey(0)
cv.destroyAllWindows()
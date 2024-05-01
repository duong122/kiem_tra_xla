import cv2 as cv
import numpy as np

def resize_image(img, h_new):
    h_old, w_old = img.shape[:2]
    w_new = int(w_old * h_new / h_old)
    resize_img = cv.resize(img, (w_new, h_new), interpolation=cv.INTER_AREA)
    cv.imwrite("resize_img.jpg", resize_img)

I = cv.imread(r"D:\Image_processing\test_contour.png", cv.IMREAD_COLOR)
h, w = I.shape[:2]
print("Ty le giua gia tri chieu cao va chieu rong cua anh la: ", h / w)
cv.imshow("origin image", I)
print("height = ", h, "\n width = ", w)

resize_image(I, 256)
resizeImage = cv.imread("resize_img.jpg")
print(resizeImage.shape)
cv.imshow("anh sau khi da hieu chinh ", resizeImage)


Ihsv = cv.cvtColor(I, cv.COLOR_BGR2HSV)
h, s, v = cv.split(Ihsv)
cv.imshow("Kenh mau s cua anh Ihsv la ", s)


new_s = cv.medianBlur(s, 9)
cv.imshow("kenh mau s sau khi loc median ", new_s)














cv.waitKey(0)
cv.destroyAllWindows()
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def resize_image(img, h_new):
    h_old, w_old = img.shape[:2]
    w_new = int(w_old * h_new / h_old)
    resize_img = cv.resize(img, (w_new, h_new), interpolation=cv.INTER_AREA)
    cv.imwrite("resize_img.jpg", resize_img)

# 1. Hien thi ty le do cao va do rong cua anh I doc vao
I = cv.imread(r"D:\Image_processing\ex7.png", cv.IMREAD_COLOR)
h, w = I.shape[:2]
print("Ty le giua gia tri chieu cao va chieu rong cua anh la: ", h / w)
cv.imshow("origin image", I)
print("height = ", h, "\n width = ", w)

# 2. Hieu chinh lai anh I voi do cao moi la 256, anh giu nguyen ty le so voi anh goc, duoc anh moi I2
#    hien thi anh I2
resize_image(I, 256)
resizeImage = cv.imread("resize_img.jpg")
print(resizeImage.shape)
cv.imshow("anh sau khi da hieu chinh ", resizeImage)

# 3. Chuyen doi anh I sang anh HSV. Hien thi kenh S cua anh HSV
Ihsv = cv.cvtColor(I, cv.COLOR_BGR2HSV)
h, s, v = cv.split(Ihsv)
cv.imshow("Kenh mau s cua anh Ihsv la ", s)

# 4. Lam tron kenh S cua anh Ihsv voi bo loc median kich thuoc cua so 3 * 3, bien doi nguoc anh Ihsv ve anh
#    mau RGB duoc anh I3. Hien thi anh I3
new_s = cv.blur(s, [5,5])
cv.imshow("kenh mau s sau khi loc median ", new_s)


# 5. Xác định histogram của kênh S. Vẽ histogram
hist = cv.calcHist([Ihsv], [1], None, [256], [0, 256])
plt.plot(hist, color='k')
plt.show()

# 6. Cân bằng histogram kênh S của ảnh Ihsv. Biến đổi ngược ảnh Ihsv về biểu diễn màu RGB được ảnh I4, hiển
#    thi ảnh I4
new_s = cv.equalizeHist(s)
new_ihsv = cv.merge([h, new_s, v])
new_rgb = cv.cvtColor(new_ihsv, cv.COLOR_HSV2RGB)
cv.imshow("new_rgb image", new_rgb)


cv.waitKey(0)
cv.destroyAllWindows()
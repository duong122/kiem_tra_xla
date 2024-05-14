import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc ảnh màu vào ma trận I. Hiển thị ma trận I
I = cv.imread(r"D:\Image_processing\Ice_bear.jpg", cv.IMREAD_COLOR)
cv.imshow("anh mau doc vao la ", I)

# 2. Chuyển ảnh sang biểu diễn HSV được ma trận Ihsv. Hiển thị kênh h của Ihsv. Xác định giá trị mức sáng lớn nhất
#    của kênh S của ảnh HSV
Ihsv = cv.cvtColor(I, cv.COLOR_BGR2HSV)
h, s, v = cv.split(Ihsv)
cv.imshow("Kenh mau h cua Ihsv la", h)
max_s = s.max()
print("Gia tri muc sang lon nhat cua kenh s la: ", max_s)

# 4. Làm trơn ảnh kênh S của Ihsv theo bộ lọc median kích thước cửa sổ lân cận là 7 * 7 được ảnh Is.
#     hiển thị ảnh Is
Is = cv.medianBlur(s, 49)
cv.imshow("Kenh mau s sau khi lam tron la", Is)

# 5. Nhị phân hóa ảnh nghịch đảo của ảnh Is theo ngưỡng Otsu được ảnh Ib. Hiển thị ảnh Ib
Is_2 = 255 - Is
_, Ib = cv.threshold(Is_2, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("anh sau khi lay nguong otsu la", Ib)


# 3. Xác định và vẽ histogram kênh v cua anh Ihsv
# histogram = cv.calcHist(Ihsv, [2], None, [256], [0, 256])
# plt.plot(histogram, color='k')
plt.hist(v.ravel(), 256, (0, 256))
plt.show()

# 6. Xác định đường Contour có diện tích lớn nhất của ảnh Ib. Vẽ đường contour trên ảnh gốc I, hiển thị ảnh I
contours, hierarchy = cv.findContours(Ib, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
max_contour = contours[0]
max_area = cv.contourArea(max_contour)
for contour in contours:
    area = cv.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

cv.drawContours(I, [max_contour], -1, (0, 255, 255), 3)
cv.imshow("anh sau khi ve duong contour lon nhat la", I)



cv.waitKey(0)
cv.destroyAllWindows()


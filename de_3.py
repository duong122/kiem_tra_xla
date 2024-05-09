import cv2 as cv
import numpy as np

I = cv.imread(r"D:\Image_processing\color_split.png", cv.IMREAD_COLOR)
cv.imshow("origin image", I)
gray_scale = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
print("gray scale", gray_scale)
Ihsv = cv.cvtColor(I, cv.COLOR_BGR2HSV)
h, s, v = cv.split(Ihsv)
cv.imshow("kenh mau v", v)
v = np.uint8(v)
max_v = v.max()
print("Muc sang lon nhat cua kenh mau v la: ", max_v)

kernel = np.ones((5, 5), np.float64) / 25
Is = cv.filter2D(I, -1, kernel)
cv.imshow("anh sau khi loc trung binh la ", Is)

Is_2 = 255 - Is
print(Is_2)
cv.imshow("Is_2", Is_2)
gray_scale = cv.cvtColor(Is_2, cv.COLOR_BGR2GRAY)
_, Ib = cv.threshold(gray_scale, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("Ib", Ib)

# 5. Xác định đường contour có diện tích lớn nhất của ảnh Ib. Vẽ đường contour trên ảnh gốc I
#  và hiển thị ảnh I
contours, hierarchy = cv.findContours(Ib, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
area_contour_max = 0
result_contour = 0
for contour in contours:
    area = cv.contourArea(contour)
    if area > area_contour_max:
        area_contour_max = area
        result_contour = contour
cv.drawContours(I, [result_contour], -1, (0, 255, 255), 3)
cv.imshow("Anh sau khi duoc ve them contour la: ", I)

# 6. Tăng độ sáng của kênh V của ảnh Ihsv bằng phương phap giãn mức xám. Biến đổi ngược ảnh Ihsv về
# biểu diễn màu RGB được ảnh I. Hiển thị lại ảnh I.










cv.waitKey(0)
cv.destroyAllWindows()
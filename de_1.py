import cv2 as cv
import numpy as np

#  đọc vào một ảnh màu
origin_image = cv.imread(r"D:\Image_processing\Ice_bear.jpg")
cv.imshow('ex1', origin_image)
# cv.waitKey(0)

blue, green, red = cv.split(origin_image)
# cv.imshow("blue", blue)
# cv.imshow("green", green)
# cv.imshow("red", red)
# cv.waitKey(0)


h, w = origin_image.shape[:2]
print(h, w)
Ig = np.zeros((h, w), np.uint8)
for i in range(h):
    for j in range(w):
        Ig[i, j] = int(0.39*red[i, j] + 0.5*green[i, j] + 0.11*blue[i, j])
        print(Ig[i, j])
cv.imshow('gray scale img', Ig)

img_gaussian = cv.GaussianBlur(Ig, (3,3), 0)
img_canny = cv.Canny(img_gaussian, 300, 200)

cv.imshow("image canny", img_canny)

y = 160
x = 326
if img_canny[y, x] == 255:
    print("Diem anh da cho la diem bien")
else:
    print("Diem anh da cho khong phai la diem bien")

# nhi phan anh thu duoc theo nguong otsu
_, Ib = cv.threshold(Ig, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imshow('Ib image', Ib)

contours, hierarchy = cv.findContours(img_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cv.imshow('Canny Edges After Contouring', img_canny)
print("Number of Contours found = " + str(len(contours)))

# len_contour = len(contours)
# for i in range(len_contour):
#     print("Do lon la: ", contours[i].shape)

# area = [cv.contourArea(c) for c in contours]
max_area_matrix = max(contours, key=cv.contourArea)
max_area = cv.contourArea(max_area_matrix)
result_contour = [contour for contour in contours if cv.contourArea(contour) > (max_area / 5)]
print("max area", max_area)

cv.drawContours(origin_image, result_contour, -1, (0, 255, 255), 4)
cv.imshow("Contour", origin_image)

cv.drawContours(origin_image, contours, -1,(0, 255, 0),3)
cv.imshow('Contours', origin_image)
cv.waitKey(0)
cv.destroyAllWindows()



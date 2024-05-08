import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. Doc vao anh mau vao bien ma tran I va hien thi anh vua doc vao
I = cv.imread(r"D:\Image_processing\Ice_bear.jpg")
img = I
blue, green, red = cv.split(I)

cv.imshow("blue", blue)
gray = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)

# 2. Chuyen anh sang he mau hsv va hien thi kenh h cua anh
Ihsv = cv.cvtColor(I, cv.COLOR_BGR2HSV)
cv.imshow("Ihsv", Ihsv)
h, s, v = cv.split(Ihsv)
cv.imshow("h", h)

# 2. Xac dinh gia tri muc sang trung binh cua kenh S
height, width = Ihsv.shape[:2]
sum = Ihsv.sum()
avg_sum = sum / (height * width)
print("Gia tri muc xam trung binh cua kenh S la: ", avg_sum)

# 4. Lam tron anh kenh V theo bo loc trung binh cong, kich thuoc cua so lan can la 3*3 duoc anh
# Is hien thi anh Is
Is = cv.blur(v, (3, 3))
cv.imshow("kenh v sau khi duoc lam tron la: ", Is)

# 5. Nhi phan hoa anh Is theo nguong Otsu duoc anh nhi phan Ib. Hien thi anh Ib
_, Ib = cv.threshold(Is, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("Anh sau khi nhi phan hoa theo nguong otsu la", Ib)

# 6. Xac dinh duong contour co ty le giua chu vi va dien tich lon nhat cua anh Ib. Ve duong contour do
#    tren anh goc I
contours, hierarchy = cv.findContours(Ib, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
chu_vi = []
for cnt in contours:
    tmp = cv.arcLength(cnt, True)
    chu_vi.append(tmp)

dien_tich = []
for contour in contours:
    tmp = cv.contourArea(contour)
    dien_tich.append(tmp)

chu_vi = np.array(chu_vi)
dien_tich = np.array(dien_tich)
result = chu_vi / dien_tich
max_result = result.argmax()

cv.drawContours(Ib, contours, -1, (0, 255, 255), 3)
cv.imshow("Anh sau khi tim duoc contour co ty le lon nhat la", Ib)


cv.waitKey(0)
cv.destroyAllWindows()

# 3. Xac dinh va ve historgam cua kenh S
def compute_hist(img):
    hist = np.zeros(256, np.uint8)
    h, w = I.shape[:2]
    for i in range(h):
        for j in range(w):
            hist[I[i, j]] += 1
    return hist

def equal_hist(hist):
    cumulator = np.zeros_like(hist, np.float64)
    for i in range(1, len(cumulator)):
        cumulator[i - 1] = hist[:i].sum()
    print(cumulator)
    new_hist = ((cumulator - cumulator.min()) / (cumulator.max() - cumulator.min())) * 255
    new_hist = np.uint8(new_hist)
    return new_hist

hist = compute_hist(img).ravel()
new_hist = equal_hist(hist)

h, w = img.shape[:2]
for i in range(h):
    for j in range(w):
        img[i, j] = new_hist[img[i, j]]

fig = plt.figure()
ax = plt.subplot(121)
plt.imshow(I, cmap='gray')
plt.subplot(122)
plt.imshow(img, cmap='gray')
plt.show()

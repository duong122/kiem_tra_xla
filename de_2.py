import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

I = cv.imread(r"D:\Image_processing\color_split.png")
img = I
blue, green, red = cv.split(I)

cv.imshow("blue", blue)
gray = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)

Ihsv = cv.cvtColor(I, cv.COLOR_BGR2HSV)
cv.imshow("Ihsv", Ihsv)
h, s, v = cv.split(Ihsv)
cv.imshow("h", h)

height, width = Ihsv.shape[:2]
sum = 0
for i in range(height):
    for j in range(width):
        sum += Ihsv[i, j, 1]
avg_sum = sum / (height * width)
print("Gia tri muc xam trung binh cua kenh S la: ", avg_sum)

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


cv.waitKey(0)
cv.destroyAllWindows()





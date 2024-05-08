import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

I = cv.imread(r"D:\Image_processing\Ice_bear.jpg", cv.IMREAD_COLOR)
cv.imshow("anh mau doc vao la ", I)

Ihsv = cv.cvtColor(I, cv.COLOR_BGR2HSV)
h, s, v = cv.split(Ihsv)
cv.imshow("Kenh mau h cua Ihsv la", h)
max_s = s.max()
print("Gia tri muc sang lon nhat cua kenh s la: ", max_s)


Is = cv.medianBlur(s, 49)
cv.imshow("Kenh mau s sau khi lam tron la", Is)

Is_2 = 255 - Is
_, Ib = cv.threshold(Is_2, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("anh sau khi lay nguong otsu la", Ib)



# histogram = cv.calcHist(Ihsv, [2], None, [256], [0, 256])
# plt.plot(histogram, color='k')
plt.hist(v.ravel(), 256, (0, 256))
plt.show()































cv.waitKey(0)
cv.destroyAllWindows()
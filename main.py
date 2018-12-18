import numpy as np
import cv2

img = cv2.imread('./input/trang1.png')
heigh, width = img.shape[:2]

cv2.line(img, (0, 46), (width, 46), (40, 150, 48), 1)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

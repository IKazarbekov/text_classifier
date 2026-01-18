import cv2
import numpy as np

im = cv2.imread("../data/raw/2025-12-13-111040.jpg")
w, h = im.shape[:2]

diff = 500
mask = np.zeros((w + 2,h + 2), np.uint8)
cv2.floodFill(im, mask, (900,300),(250,250,0), diff, diff)

cv2.imshow("flood fill",im)
cv2.waitKey()
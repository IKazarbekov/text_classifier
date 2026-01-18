import cv2
import numpy as np
from PIL import Image

im = cv2.imread("../../data/processed/character_в_1.png")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# for crop image
flag = False
for i in range(len(gray)):
    for j in range(len(gray[i])):
        if gray[i][j] <= 254:
            maximum_top = i
            flag = True
            break
    if flag:
        flag = False
        break
for i in range(len(gray[0])):
    for j in range(len(gray)):
        if gray[j][i] <= 254:
            maximum_left = i
            flag = True
            break
    if flag:
        flag = False
        break
for i in range(len(gray[0]) - 1, 0, -1):
    for j in range(len(gray)):
        if gray[j][i] <= 254:
            maximum_right = i
            flag = True
            break
    if flag:
        flag = False
        break
for i in range(len(gray) - 1, 0, -1):
    for j in range(len(gray[i])):
        if gray[i][j] <= 254:
            maximum_down = i
            flag = True
            break
    if flag:
        flag = False
        break
# ! Change PIL on cv2
pil_im = Image.fromarray(gray)
pil_im = pil_im.crop((maximum_left, maximum_top, maximum_right, maximum_down))
crop = np.array(pil_im)

bottom = 200 - len(crop)
right = 200 - len(crop[0])
big = cv2.copyMakeBorder(
    crop,
    0, bottom, 0, right,
    cv2.BORDER_CONSTANT,
    value=[255,255,255]
)
# ! Change if may on cv2
def invert_image(im):
    invert = np.array(im)
    for i in range(len(invert)):
        for j in range(len(invert[0])):
            if invert[i][j] > 200:
                invert[i][j] = 0
            else:
                invert[i][j] = 255
    return invert
invert = invert_image(big)

def show(title, im):
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
show("original image", im)
show("gray image", gray)
show("crop image", crop)
show("big image", big)
show("invert image", invert)
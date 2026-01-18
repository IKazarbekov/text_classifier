import cv2
import numpy as np
from sklearn.linear_model import Perceptron
from PIL import Image

def show(title, im, ms=100):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, im)
    cv2.waitKey(ms)
    cv2.destroyAllWindows()

def processed_big_image(im):
    assert im.shape[2]==3 , "ImageErrorSize"
    blur = cv2.medianBlur(im, 1)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # ! Change if may on cv2
    def invert_image(im):
        invert = np.array(im)
        for i in range(len(invert)):
            for j in range(len(invert[0])):
                if invert[i][j] > 150:
                    invert[i][j] = 0
                else:
                    invert[i][j] = 255
        return invert
    invert = invert_image(gray)

    # for crop image
    flag = False
    for i in range(len(invert)):
        for j in range(len(invert[i])):
            if invert[i][j] > 254:
                maximum_top = i
                flag = True
                break
        if flag:
            flag = False
            break
    for i in range(len(invert[0])):
        for j in range(len(invert)):
            if invert[j][i] > 254:
                maximum_left = i
                flag = True
                break
        if flag:
            flag = False
            break
    for i in range(len(invert[0]) - 1, 0, -1):
        for j in range(len(invert)):
            if invert[j][i] > 254:
                maximum_right = i
                flag = True
                break
        if flag:
            flag = False
            break
    for i in range(len(invert) - 1, 0, -1):
        for j in range(len(invert[i])):
            if invert[i][j] > 254:
                maximum_down = i
                flag = True
                break
        if flag:
            flag = False
            break
    # ! Change PIL on cv2
    pil_im = Image.fromarray(invert)
    pil_im = pil_im.crop((maximum_left, maximum_top, maximum_right, maximum_down))
    crop = np.array(pil_im)

    bottom = 100
    right = 100
    big = cv2.copyMakeBorder(
        crop,
        0, bottom, 0, right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    return big
im = cv2.imread("../../data/raw/good_privet.png")
pim = processed_big_image(im)

points = list()
for y in range(len(pim)):
    for x in range(len(pim[y])):
        if pim[y][x] > 150:
            if (len(points) != 0 and x < points[-1][0] and abs(x - points[-1][0]) > 20): # if dinstance long
                points.clear()
            if len(points) == 0 or x < points[-1][0]:
                points.append((x, y))
                break
y = points[0][1]
x = points[-1][0]

print(points)
print(y, x)
show("privet", pim, ms=0)

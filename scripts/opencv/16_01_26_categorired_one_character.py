import cv2
import numpy as np
from sklearn.linear_model import Perceptron
from PIL import Image

def processed_image(im):
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

    bottom = 200 - len(crop)
    right = 200 - len(crop[0])
    big = cv2.copyMakeBorder(
        crop,
        0, bottom, 0, right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    return big

im_v1 = cv2.imread("../../data/processed/character_в_1.png")
im_v2 = cv2.imread("../../data/processed/character_в_2.png")
im_v3 = cv2.imread("../../data/processed/character_в_3.png")
im_v4 = cv2.imread("../../data/processed/character_в_4.png")
im_p1 = cv2.imread("../../data/processed/character_п_1.png")

pim_v1 = processed_image(im_v1)
pim_v2 = processed_image(im_v2)
pim_v3 = processed_image(im_v3)
pim_v4 = processed_image(im_v4)
pim_p1 = processed_image(im_p1)

def show(title, im):
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
show("B1", pim_v1)
show("B2", pim_v2)
show("B3", pim_v3)
show("B4", pim_v4)
show("P1", pim_p1)

def processed_image_for_pnn(im):
    assert im.shape == (200, 200), "ErrorImageSize"
    return im.reshape(40000)
ppnim_v1 = processed_image_for_pnn(pim_p1)
ppnim_v2 = processed_image_for_pnn(pim_v2)
ppnim_v3 = processed_image_for_pnn(pim_v3)
ppnim_v4 = processed_image_for_pnn(pim_v4)
ppnim_p1 = processed_image_for_pnn(pim_v1)
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit([ppnim_v1, ppnim_v3, ppnim_v4, ppnim_p1], ["в","в","в", "п"])
print(ppn.predict([ppnim_v2]))
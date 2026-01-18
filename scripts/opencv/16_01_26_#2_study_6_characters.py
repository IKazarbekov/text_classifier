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
def show(title, im):
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, im)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
def processed_image_for_pnn(im):
    assert im.shape == (200, 200), "ErrorImageSize"
    return im.reshape(40000)

paths_value = {
    "../../data/processed/character_в_1.png" : "в",
    "../../data/processed/character_в_2.png" : "в",
    "../../data/processed/character_в_3.png": "в",
    "../../data/processed/character_в_4.png": "в",
    "../../data/processed/character_п_1.png": "п",
    "../../data/processed/character_р_1.png": "р",
    "../../data/processed/character_е_1.png": "е",
    "../../data/processed/character_т_1.png": "т",
    "../../data/processed/character_и_1.png": "и",
}
images_pnn = list()
values_pnn = list()
for path in paths_value:
    im = processed_image(cv2.imread(path))
    show(paths_value[path], im)
    im_pnn = processed_image_for_pnn(im)
    images_pnn.append(im_pnn)
    values_pnn.append(paths_value[path])

ppn = Perceptron()
ppn.fit(images_pnn, values_pnn)

print(ppn.predict([images_pnn[5]]))
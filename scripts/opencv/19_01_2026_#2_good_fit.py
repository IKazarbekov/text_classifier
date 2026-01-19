import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron


def show(title, im, ms=100):
    if ms < 0:
        return
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, im)
    cv2.waitKey(ms)
    cv2.destroyAllWindows()
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
    #pil_im = Image.fromarray(invert)
    #pil_im = pil_im.crop((maximum_left, maximum_top, maximum_right, maximum_down))
    #crop = np.array(pil_im)
    crop = invert[maximum_top:maximum_down, maximum_left:maximum_right]

    bottom = 200 - len(crop)
    right = 200 - len(crop[0])
    big = cv2.copyMakeBorder(
        crop,
        0, bottom, 0, right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    return big
def processed_image_for_pnn(im):
    assert im.shape == (200, 200), "ErrorImageSize"
    return im.reshape(40000)

paths_value = {
    "../../data/processed/character_в_1.png": ("в", 1),
    "../../data/processed/character_п_1.png": ("п", 1),
    "../../data/processed/character_т_7.png": ("т", 1),
    "../../data/processed/character_р_1.png": ("р", 1),
    "../../data/processed/character_е_1.png": ("е", 2),
    "../../data/processed/character_и_1.png": ("и", 1),
}
images_pnn = list()
images = list()
values_pnn = list()
weight_pnn = list()
for path in paths_value:
    im = processed_image(cv2.imread(path))
    images.append(im)
    im_pnn = processed_image_for_pnn(im)
    images_pnn.append(im_pnn)
    values_pnn.append(paths_value[path][0])
    weight_pnn.append(paths_value[path][1])

#ppn = SGDClassifier(loss="perceptron")
ppn = Perceptron(eta0=0.001)
all_images_pnn = list()
all_values_pnn = list()
# crop from 100% to 20% image and fit
for i in range(200, 20, -10):
    size = 2 * i
    images_resize = list()
    values_rename = list()
    for im in images:
        im_resize = cv2.resize(im,(size, size))
        if (size < 200):
            im_resize = cv2.copyMakeBorder(
                im_resize,
                0, 200 - size, 0, 200 - size,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
        else:
            im_resize = im_resize[0:200, 0:200]
        #im_resize = cv2.GaussianBlur(im_resize, (5, 5), 5 - (200 - i) // 40)
        show("resize", im_resize, ms=-1)
        im_resize_pnn = processed_image_for_pnn(im_resize)
        #im_resize_pnn = cv2.threshold(im_resize_pnn, 250, 255, cv2.THRESH_BINARY)
        im_resize_pnn = np.array( [x if x > 200 else -200 for x in im_resize_pnn])
        images_resize.append(im_resize_pnn)
    for value in values_pnn:
        new_name = value + "_size:" + str(size)
        print(new_name)
        values_rename.append( new_name )
    all_images_pnn.extend(images_resize)
    all_values_pnn.extend(values_rename)
ppn.fit(all_images_pnn, all_values_pnn)

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
#im = cv2.imread("../../data/raw/good_privet.png")
#im = cv2.imread("../../data/raw/good_terpit.png")
im = cv2.imread("../../data/raw/good_eralash.jpg")
pim = processed_big_image(im)

def find_begin_point(im):
    points = list()
    for y in range(len(im)):
        for x in range(len(im[y])):
            if im[y][x] > 150:
                if (len(points) != 0 and x < points[-1][0] and abs(x - points[-1][0]) > 20): # if dinstance long
                    points.clear()
                if len(points) == 0 or x < points[-1][0]:
                    points.append((x, y))
                    break
    y = points[0][1]
    x = points[-1][0]
    return x, y

x, y = find_begin_point(pim)
print(f"begin point:{y}:{x}")
show("privet", pim)

def crop_image(im, x, y, size):
    # ! Change pil on cv2 if may
    #pil_im = Image.fromarray(pim)
    #pil_im = pil_im.crop((x, y, x + size, y + size))
    return im[y:y+size, x:x+size]

answers = list()
for i in range(50, 150, 10):
    crop = crop_image(pim, x, y, i)

    im_resize = cv2.resize(crop, (200, 200))
    #im_resize = cv2.GaussianBlur(im_resize, (101, 101), 5 - i // 50)
    im_ppn = processed_image_for_pnn(im_resize)

    answer = ppn.predict([im_ppn])[0][0]
    answers.append(answer)
    print("asnwer perseptron" , answer)
    show("im_ppn", im_resize, ms=0)
#end_answer = max(set(answers), answers.count)
#print("End answer: ", end_answer)

coefs = ppn.coef_
print("======= MODEL ========")
print(f"count coef: {coefs.shape[0]}")
print(f"shape coef: {coefs.shape}")
print(f"count study: {ppn.t_}")
print(f"coef type: {ppn.coef_.dtype}")
for coef in coefs:
    im = coef.reshape((200, 200))
    print(im)
    show("res", im, ms=0)
    pass
import cv2
from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt

# загрузка фото
im_num_0 = cv2.imread("../../../data/processed/number_0(0).png")
im_num_1 = cv2.imread("../../../data/processed/number_1(0).png")
im_num_2 = cv2.imread("../../../data/processed/number_2(0).png")
im_num_3 = cv2.imread("../../../data/processed/number_3(0).png")

im_num_1_1 = cv2.imread("../../../data/processed/number_1(1).png")

# обработчка фотографий
im_num_0 = cv2.cvtColor(im_num_0, cv2.COLOR_BGR2GRAY).reshape(10000)
im_num_1 = cv2.cvtColor(im_num_1, cv2.COLOR_BGR2GRAY).reshape(10000)
im_num_2 = cv2.cvtColor(im_num_2, cv2.COLOR_BGR2GRAY).reshape(10000)
im_num_3 = cv2.cvtColor(im_num_3, cv2.COLOR_BGR2GRAY).reshape(10000)

im_num_1_1 = cv2.cvtColor(im_num_1_1, cv2.COLOR_BGR2GRAY).reshape(10000)

perceptron = Perceptron(eta0=0.01, random_state=1)
perceptron.fit([im_num_0, im_num_1], [0, 1])

perceptron.partial_fit([im_num_2, im_num_3], [2,3])
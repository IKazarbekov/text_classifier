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

# создание массива признаков и меток
x = np.array([im_num_0, im_num_1, im_num_2, im_num_3])
y = np.array([0,1,2,3])
#создание и обучение модели
perceptron = Perceptron(eta0=0.01, random_state=1)
perceptron.fit(x,y)

"""
# data in perceptron
array = perceptron.coef_.reshape(100, 100, 4)
print(array)
#visual
plt.imshow(array)
plt.show()"""

# show intercept
intersept_array = perceptron.intercept_
print(intersept_array)

# answer
print(perceptron.predict([im_num_1_1]))


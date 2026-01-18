import PySimpleGUI as sg

import cv2
import numpy as np
from PIL import Image
from sklearn.externals.array_api_compat.torch import maximum

from sklearn.linear_model import Perceptron
import joblib

from prompt_toolkit.shortcuts import button_dialog
from sipbuild.generator.parser.rules import p_pod_type



def update_image(window, image_array):
    img_bytes = cv2.imencode('.png', image_array)[1].tobytes()
    window["-IMAGE-"].update(data=img_bytes)
    update_text_about_image(window, f"Размер:{len(image_array)}x{len(image_array[0])}")

def update_text_about_image(window, text):
    window["-TEXT_ABOUT_IMAGE-"].update(text)

history_images = list()

if __name__ == "__main__":

    # Модель
    ppt = Perceptron(eta0=0.01, random_state=1)

    # Интерфейс
    layout = [
        [
            sg.Text("Модель: Новая модель")
        ],
        [
            sg.Frame("Загрузка изображения", [
            [
                sg.Image(key="-IMAGE-", size=(100, 100), background_color="white"),
                sg.Text("Описание\nИзображения нет",key="-TEXT_ABOUT_IMAGE-"),
            ],
            [
                sg.Button("Загрузить", key="-LOAD_IMAGE-"),
                sg.Button("Очистить", key="-CLEAR_IMAGE-"),
                sg.Button("Сохранить", key="-SAVE_IMAGE-")
             ],
                [sg.Button("Обесцветить", key="-DELCOLOR-")],
                [sg.Button("Обрезать", key="-CROP-")],
                [sg.Button("Изменить размер на стандартный", key="-CHANGE_SIZE-")],
                [sg.Button("Отмена", key="-ABOLITION-")],
                [sg.Text("", key="-TEXT_ERROR-", text_color="red")]
        ])],
        [
            sg.Frame("Обучение модели",[
                [sg.Text("Ответ: "),sg.Input("")],
                [sg.Button("Обучить",key="-STUDY-")]
            ])
        ],
        [
            sg.Frame("Прогноз модели", [
                [sg.Button("Получить прогноз", key="-ANSWER-")],
                [sg.Text("", key="-ANSWER_TEXT-")]
            ])
        ]
    ]

    window = sg.Window("ScanText", layout)

    while True:
        event, values = window.read()
        match event:
            case sg.WIN_CLOSED:
                break
            case "-LOAD_IMAGE-":
                #file_path = sg.popup_get_file("Выберите изображение", file_types=("Images", "*.png *.jpg *.jpeg"))
                #current_image = cv2.imread(file_path)
                current_image = cv2.imread(
                    "/data/good data/character_в_1.png")

                update_image(window, current_image)
                history_images.append(current_image)
            case "-CLEAR_IMAGE-":
                current_image = None
                window["-IMAGE-"].update(data=None)
            case "-DELCOLOR-":
                current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                update_image(window, current_image)
                print(current_image.ndim)
            case "-CROP-":

                # проверка на то что изображение без цветов
                if current_image.ndim != 2:
                    window["-TEXT_ERROR-"].update("Сначало обесцветьте изображение")
                    continue

                flag = False
                for i in range(len(current_image)):
                        for j in range(len(current_image[i])):
                            if current_image[i][j] <= 254:
                                maximum_top = i
                                flag = True
                                break
                        if flag:
                            flag = False
                            break
                for i in range(len(current_image[0])):
                        for j in range(len(current_image)):
                            if current_image[j][i] <= 254:
                                maximum_left = i
                                flag = True
                                break
                        if flag:
                            flag = False
                            break
                for i in range(len(current_image[0]) - 1,0,-1):
                        for j in range(len(current_image)):
                            if current_image[j][i] <= 254:
                                maximum_right = i
                                flag = True
                                break
                        if flag:
                            flag = False
                            break
                for i in range(len(current_image) - 1, 0, -1):
                        for j in range(len(current_image[i])):
                            if current_image[i][j] <= 254:
                                maximum_down = i
                                flag = True
                                break
                        if flag:
                            flag = False
                            break

                pil_im = Image.fromarray(current_image)
                pil_im = pil_im.crop((maximum_left, maximum_top, maximum_right, maximum_down))
                current_image = np.array(pil_im)
                update_image(window, current_image)
            case "-ABOLITION-":
                current_image = history_images[len(history_images) - 1]
                update_image(window, current_image)
            case "-CHANGE_SIZE-":
                image = Image.fromarray(current_image)  # create PIL image
                image = image.resize((100, 100))        # resize to 100 x 100
                current_image = np.array(image)         # save to np array
                update_image(window, current_image)     # update image in GUI
            case "-STUDY-":
                answer = "b"
                study_image = current_image.reshape(10000)
                print(study_image.shape)

                X = np.array([study_image])
                Y = np.array([answer])

                ppt.fit(X, Y)
            case "-ANSWER-":   # get answer from model
                print("answer:", ppt.predict(current_image))
    window.close()
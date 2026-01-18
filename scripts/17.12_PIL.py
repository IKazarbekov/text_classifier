import PIL.Image
from PIL import Image

"""# поворот камеры
im = Image.open("test data/школа.jpg")

im = im.rotate(30)

im.save("file.png")"""

"""# ВСТАВКА ОБЛАСТИ
im = Image.open("test data/работа.jpg")
im1 = Image.open("test data/центр.jpg")
im1 = im1.resize((100,100))

im.paste(im1,(100,100,200,200))
im.save("file.png")"""

"""# ПОВОРОТ ОБЛАСТИ
pil_im = Image.open("../data/row data/2025-12-13-110132.jpg")

# поворот
pil_im  = pil_im.transpose(PIL.Image.ROTATE_90)

pil_im.save("file.png")"""

# ВЫРЕЗКА ОБЛАСТИ
"""# открытые
pil_im = Image.open("../data/row data/2025-12-13-110132.jpg")

# вырезка
pil_im1  = pil_im.crop((300,300,400,400))

pil_im.save("file.png")"""
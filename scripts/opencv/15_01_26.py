import cv2

#im = cv2.imread("../../data/processed/character_в_1.png")
im = cv2.imread("../../data/raw/2025-12-24-092502.jpg")
#im = cv2.imread("../../data/raw/2025-12-24-092508.jpg", cv2.IMREAD_REDUCED_COLOR_4)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

clean = cv2.medianBlur(gray, 3)

edges = cv2.Canny(clean, 50, 150)

def show(title, im):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

show("original image", im)

show("gray image", gray)

show("clean image", clean)

show("edges image", edges)

import cv2
import mahotas
import numpy

bins = 8


# Просто подсчет момента от картинки (анализируется форма)
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# Вычисление "фичи" текстуры
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick


# Вычисление "фичи" цвета
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# Данная функция выделяет на картинке диапазон из зеленых и
# голубых цветов. Объекты этого цвета становятся белыми,
# все остальное - черным. От полученной картинки подсчитывается
# момент (анализируется форма)
def fd_my_test(image):
    hsv_min = numpy.array((45, 50, 50), numpy.uint8)
    hsv_max = numpy.array((135, 255, 255), numpy.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    feature = cv2.HuMoments(cv2.moments(thresh)).flatten()
    return feature

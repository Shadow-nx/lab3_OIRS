import os
import numpy as np
from features import *
import warnings

warnings.filterwarnings('ignore')

# Обьявляем константы
NUM_TREES = 200
TEST_SIZE = 0.2
SEED = 9
TRAIN_PATH = "data/train"
TEST_PATH = "data/test"
H5_DATA = 'output/data.h5'
H5_LABELS = 'output/labels.h5'
SCORING = "accuracy"

IMAGES_PER_CLASS = 80
FIXED_SIZE = tuple((500, 500))

TRAIN_LABELS = os.listdir(TRAIN_PATH)
TRAIN_LABELS.sort()


# получаем наши фичи из изображений и преборазуем их в один вектор
def get_feature(image):
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    fv_histogram = fd_histogram(image)
    fv__my_test = fd_my_test(image)
    global_feature = np.hstack([fv_histogram, fv_hu_moments, fv_haralick, fv__my_test])
    return global_feature

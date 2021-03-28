from sklearn.preprocessing import LabelEncoder
from constants import *
import cv2
import os
import h5py
import warnings

warnings.filterwarnings('ignore')

def init():
    print(TRAIN_LABELS)
    global_features = []
    labels = []

    #цикл по папкам в дирректории train
    for training_name in TRAIN_LABELS:
        current_dir = os.path.join(TRAIN_PATH, training_name)
        images = [f for f in os.listdir(current_dir) if
                  os.path.isfile(os.path.join(current_dir, f)) and f.endswith(".jpg") or f.endswith(".png")]

        #цикл по картинкам в дирректории
        for file in images:
            file_path = os.path.join(current_dir, file)

            #читаем изображение и изменяем его размер
            image = cv2.imread(file_path)
            image = cv2.resize(image, FIXED_SIZE)

            #получаем наши фичи для конкретного изображения
            global_feature = get_feature(image)

            #обновляем список наших фич добавляя полученные из картинки
            labels.append(training_name)
            global_features.append(global_feature)

        print(f"[STATUS] processed folder: {training_name}")

    le = LabelEncoder()
    target = le.fit_transform(labels)

    #сохранение наших фич в .h5 файл
    h5f_data = h5py.File(H5_DATA, 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(global_features))

    h5f_label = h5py.File(H5_LABELS, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))

    h5f_data.close()
    h5f_label.close()

    print("[STATUS] end of training..")

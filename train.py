import h5py
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from constants import *


def training():
    if not os.path.exists(TEST_PATH):
        os.makedirs(TEST_PATH)

    # создаем список моделей машинного обучения которыми будем пользоваться
    models = [('LR', LogisticRegression(random_state=SEED)), 
              ('LDA', LinearDiscriminantAnalysis()),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier(random_state=SEED)),
              ('RF', RandomForestClassifier(n_estimators=NUM_TREES, random_state=SEED)),
              ('NB', GaussianNB()),
              ('SVM', SVC(random_state=SEED))]

    # создаем списки для результатов для трех метрик
    results_acc = []
    results_rec = []
    results_prec = []
    names = []

    # импортируем наши фичи из ранее созданных файлов h5
    h5f_data = h5py.File(H5_DATA, 'r')
    h5f_label = h5py.File(H5_LABELS, 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels = np.array(global_labels_string)

    h5f_data.close()
    h5f_label.close()

    # вывод обьемов векторов
    print(f"[STATUS] features shape: {global_features.shape}")
    print(f"[STATUS] labels shape: {global_labels.shape}")

    print("[STATUS] training started...")

    # разделение фич и лейблов на обучающие и тестовые в соответствие с константами
    (x_train, x_test, y_train, y_test) = train_test_split(np.array(global_features),
                                                          np.array(global_labels),
                                                          test_size=TEST_SIZE,
                                                          random_state=SEED)

    print("[STATUS] separation train and test data...")
    print(f"Train data  : {x_train.shape}")
    print(f"Test data   : {x_test.shape}")
    print(f"Train labels: {y_train.shape}")
    print(f"Test labels : {y_test.shape}")

    # для каждой из моделей нейронных сетей обучаем испльзуя 3 метрики по очереди
    for name, model in models:
        # используем кроссваледацию для наших трех метрик по очереди
        # принимают на вход модель, признаки и классы, метрику для оценки
        # делает фит и предикт на x и y, делит на cv баккетов и на каждой из групп обучается и возвращается результат
        # возвращает массив с результатом обучения на каждой группе
        cv_results = cross_val_score(model, x_train, y_train, cv=3, scoring=SCORING, n_jobs=-1)
        print(type(cv_results))
        cv_results_rec = cross_val_score(model, x_train, y_train, cv=3, scoring="precision_macro", n_jobs=-1)
        cv_results_prec = cross_val_score(model, x_train, y_train, cv=3, scoring="recall_macro", n_jobs=-1)

        model.fit(x_train, y_train)
        print(classification_report(y_test, model.predict(x_test), target_names=TRAIN_LABELS))

        # сохраняем результат для каждой из трех метрик
        results_acc.append(cv_results)
        results_rec.append(cv_results_rec)
        results_prec.append(cv_results_prec)

        names.append(name)
        print(f'{name}: {cv_results.mean()}')

    # вывод ящиков с усами для каждого метода для метрики accuracy
    fig = plt.figure()
    fig.suptitle(SCORING)
    ax = fig.add_subplot(111)
    ax.grid()
    plt.boxplot(results_acc)
    ax.set_xticklabels(names)
    plt.savefig('data/result/box1.png')

    # вывод ящиков с усами для каждого метода для метрики precision
    fig = plt.figure()
    fig.suptitle('precision')
    ax = fig.add_subplot(111)
    ax.grid()
    plt.boxplot(results_rec)
    ax.set_xticklabels(names)
    plt.savefig('data/result/box2.png')

    # вывод ящиков с усами для каждого метода для метрики recall
    fig = plt.figure()
    fig.suptitle('recall')
    ax = fig.add_subplot(111)
    ax.grid()
    plt.boxplot(results_prec)
    ax.set_xticklabels(names)
    plt.savefig('data/result/box3.png')

    return x_train, y_train

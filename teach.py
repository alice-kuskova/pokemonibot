# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:08:59 2019

@author: Alisa Kuskova
"""

#импорт библиотек
from os import listdir, makedirs #работа с файловой системой
from os.path import join, exists, isdir 

from time import time #работа со временем

import numpy as np #работа с числовыми массивами и матрицами

#функции библиотеки sklearn для работы с моделью классификации
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

#работа с изображениями
from PIL import Image

#более удобная работа со словарями
from bunch import Bunch

#наш модуль обработки изображений
import image_modify as im

#перемнные для хранения настроек программы 
MODEL_PATH = './trained_model' #папка для сохранения модели 
IMAGES_PATH = './pokemon_processed' #путь к классифицированным образцами изображений
MODEL_FILE = 'pokemon_model.pkl' #файл для сохранения обученного классификатора svc/svm
MODEL_PCA_FILE = 'pokemon_model.pca' #файл для сохранения параметров метода pca 
MODEL_CATEGORIES_NAMES_FILE = 'pokemon_model.names' # файл для сохранения списка соответствия номера категории и имени покемона 

IMAGE_ARRAY_SIZE = 2048 #размерность вектора, образованного из обработанного изображения 

#Функция, обрабатывающая классифицированные образцы изображений.
#возвращает словарь, содержащий данные, необходимые для обучения модели: 
#набор многомерных векторов для каждой категории покемона, список категорий покемонов
def get_train_images():
    #собирает список путей к образцам изображений
    pokemon_names, pokemon_paths = [], []
    for pokemon_name in sorted(listdir(IMAGES_PATH)):
        folder = join(IMAGES_PATH, pokemon_name)
        if not isdir(folder):
            continue
        paths = [join(folder, f) for f in listdir(folder) if (not isdir(join(folder, f)) & (f != '.DS_Store'))]
        n_images = len(paths)
        pokemon_names.extend([pokemon_name] * n_images)
        pokemon_paths.extend(paths)
        view_paths = [f for f in listdir(folder) if isdir(join(folder, f))]
        for pokemon_view in view_paths:
            paths = [join(folder, pokemon_view, f) for f in listdir(join(folder, pokemon_view)) if f != '.DS_Store']
            n_images = len(paths)
            pokemon_names.extend([pokemon_name + "@" + pokemon_view] * n_images)
            pokemon_paths.extend(paths)

    n_pokemon = len(pokemon_paths)
    target_names = np.unique(pokemon_names)
    target = np.searchsorted(target_names, pokemon_names)

    #считывает изображения в память и преобразует в многомерный вектор
    pokemons = np.zeros((n_pokemon, IMAGE_ARRAY_SIZE), dtype=np.float32)
    for i, pokemon_path in enumerate(pokemon_paths):
        img = Image.open(pokemon_path)
        pokemon = im.image_to_8bit_array(img)
        pokemons[i, ...] = pokemon

    #случайным образом перемешивает изображения по каждому покемону 
    indices = np.arange(n_pokemon)
    np.random.RandomState(42).shuffle(indices)
    pokemons, target = pokemons[indices], target[indices]
  
    return Bunch(data=pokemons, images=pokemons,
                 target=target, target_names=target_names,
                 DESCR="Pokemon dataset")


#функция сохранения обученной модели в файлы на диске для дальнейшего повторного использования
def save_model(path, filename, pca_filename, categories_names, cat_filename):
    if not exists(path):
        makedirs(path)

    joblib.dump(clf, join(path, filename)) 
    joblib.dump(pca, join(path, pca_filename)) 
    joblib.dump(categories_names, join(path, cat_filename))
    


#основное тело программы 
np.random.seed(30) #инициализация генератора псевдослучайных чисел
t0 = time() #запоминает текущее время
t = time()

print("Запуск обучения")
pokemon = get_train_images()
print(round(time() - t0, 2), "сек занял сбор и преобразование образцов изображений")

x = pokemon.data #данные покемонов
y = pokemon.target #категории покемонов
n_classes = pokemon.target_names.shape[0] #количество категорий покемонов

#позволяет разбить данные на обучающие и тестовые
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

###############################################################################
#применение метода pca
n_components = 0.8 # 80%
t0 = time()
pca = PCA(n_components=n_components, whiten=True).fit(x_train)
print(round(time() - t0, 2), "сек занял расчет количества главных компонентов")
print ("Количество главных компонентов векторов для 80% точности:", pca.n_components_)

print ("Идет процесс проецирования полных векторов изображений на оси главных компонентов")
t0 = time()
x_train_pca = pca.transform(x_train) #осуществляет упрощение векторов 
print(round(time() - t0, 2), "сек занял расчет проекций главных компонентов векторов")

###############################################################################
# обучение модели методом svm/svc 
param_grid = [
        {'kernel': ['rbf'], 'C': [1]},
]
t0 = time()

#подбирает лучшие параметры для расчета моделей 
#(probability - включает расчет вероятности принадлежности к каждому классу, 
#param_grid - параметры поиска параметров поиска (шаг, функция, пр)
clf = GridSearchCV(SVC(class_weight='balanced', probability=True), param_grid, cv=5, n_jobs=-1)

clf = clf.fit(x_train_pca, y_train) #само обучение модели с найденными параметрами
print("Найдены наилучшие параметры модели:")
print(clf.best_estimator_)

print(round(time() - t0, 2), "сек заняло обучение модели")
t0 = time()

#сохраняет модель на диск 
save_model(MODEL_PATH, MODEL_FILE, MODEL_PCA_FILE, pokemon.target_names, MODEL_CATEGORIES_NAMES_FILE)


# #############################################################################
# проверка модели
print("Запуск тестирования качества распознавания на тестовых образцах")
t0 = time()
x_test_pca =  pca.transform(x_test) #упрощает тестовые данные методом pca
y_pred = clf.predict(x_test_pca) #осуществляет распознавание тестовых данных
print(round(time() - t0, 2), "сек заняло тестирование модели")

#вывод результатов проверки
print(classification_report(y_test, y_pred, target_names=pokemon.target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

print("Обучение завершено. Всего потребовалось ", round(time() - t, 2), "сек")

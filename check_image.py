# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:08:59 2019

@author: Alisa Kuskova
"""
from os.path import join
from time import time

import numpy as np

from PIL import ImageOps

import image_modify as im
from sklearn.externals import joblib


IMAGES_PATH = './check_image'
MODEL_FILE = 'pokemon_model.pkl'
MODEL_PATH = './trained_model'
MODEL_PCA_FILE = 'pokemon_model.pca'
MODEL_CATEGORIES_NAMES_FILE = 'pokemon_model.names'
IMAGE_ARRAY_SIZE = 2048 #размерность вектора, образованного из обработанного изображения 


def load_model(path, filename):
    return joblib.load(join(path, filename))

def get_image_array(image):
    x = im.image_to_8bit_array(image)
    return x

def load_image(image_path):
    return im.load_and_modify_image(image_path)
    
def mirror_image(image):
    return ImageOps.flip(image)

def get_best_variants(proba):
    max = np.max(proba)
    min = 0.8 * max
    i = 0
    result = []
    for y in range(len(proba)):
        if proba[y] >= min:
            result.append([y, proba[y]])
            i = i + 1
    return result
            
def check_file(filename):
   
    # Проверяем две версии файла - исходную и зеркально отраженную
    x_test_array = np.zeros((2, IMAGE_ARRAY_SIZE), dtype=np.float32)
    
    img = load_image(filename)
    x_test_array[0, ...] = get_image_array(img)
    x_test_array[1, ...] = get_image_array(mirror_image(img))
        
    x_test_pca = pca.transform(x_test_array.reshape(len(x_test_array), -1))
    y_pred_proba = clf.predict_proba(x_test_pca)
    
    if np.max(y_pred_proba[0]) > np.max(y_pred_proba[1]):
        best_variants = get_best_variants(y_pred_proba[0])
    else:
        best_variants = get_best_variants(y_pred_proba[1])
    
    result = []
    for v in best_variants:
        result.append([target_names[v[0]], round(v[1] * 100, 2)])
    result.sort(key=lambda x: x[1], reverse=True)
    return result                   
    
    
np.random.seed(3)
t0 = time()
t = time()

clf = load_model(MODEL_PATH, MODEL_FILE)
pca = load_model(MODEL_PATH, MODEL_PCA_FILE)
target_names = load_model(MODEL_PATH, MODEL_CATEGORIES_NAMES_FILE)

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:43:50 2019

@author: Alisa Kuskova
"""

#подключение доп библиотек
from PIL import Image, ImageOps #работа с изображениями
import numpy as np              #работа с числовыми матрицами
from skimage.feature import hog #HOG преобразование изображения


W = 256   #ширина обработанного изображения в пикселях
H = 256   #высота обработанного изображения в пикселях

#функция приведения изображения к единому размеру и формату
#path - строка, указывающая путь к обрабатываемому изображению
def load_and_modify_image(path):  
    img = Image.open(path).convert(mode='RGB')#открывается изображение, приводится к rgb формату
    img = ImageOps.pad(img, (W, H))           #изменяет размер изображения в соответствии с заданными измерениями
    return img                                #возвращает преобразованное изображение


#функция преобразования изображения в двухмерный массив, содержащий один байт на кажддый пиксель
#image - преобразуемое изображение 
def image_to_8bit_array(image):
    
    #Преобразование изображения в массив векторов, указывающих направления изменения яркости
    #для выделения границ и контрастности контуров на изображении
    #при помощи HOG (Histogram of Oriented Gradients) - гистограмма направленных градиентов
    hog_array = hog(np.asarray(image), orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=False, multichannel=True, 
                    block_norm='L2-Hys')
    return hog_array 
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:16:51 2019

@author: Alisa Kuskova
"""

from os import listdir, makedirs         #функции для списка файлов и создания директорий
from os.path import join, exists, isdir  #составление пути к файлу, проверка существования файла, проверка принадлежности к директории или файлу
from shutil import rmtree                #функция удаления директории, включая вложенные поддиректории и файлы
import image_modify as im                #функция приведения изображений к общему виду и формату

INPUT_PATH = './pokemon'                #путь к необработанным изображениям
OUTPUT_PATH = './pokemon_processed'     #путь для сохранения обработанных изображений 

#функция рекурсивной обработки изображений в указанной папке 
#input - путь к папке с необработанными изображениями, 
#output - путь для сохранения обработанных изображений
def process_folder(input, output):
    i = 1 #счетчик изображений
    for f in listdir(input): #циклом перебирает все названия файлов и подпапок
        old_path = join(input, f)
        new_path = join(output, f)
        if isdir(old_path):
            if not exists(new_path): #проверка на существование подпапки с обработанными изображениями этого класса
                makedirs(new_path)   #создание такой подпапки
            
            process_folder(old_path, new_path) #когда встречает подпапки, рекурсивно их обрабатывает
            continue
        try: #try - в случае ошибки программа не рухнет и пойдет дальше
            img = im.load_and_modify_image(old_path)
            new_path = join(output, str(i)) + '.jpg' #путь и имя обработанного изображения
            i = i + 1
            img.save(new_path) #обработанное изображение сохраняется

        except: #при возникновении ошибки выдается сообщение
           print("error in file",old_path) 
      

#тело программы
if exists(OUTPUT_PATH): 
    rmtree(OUTPUT_PATH) #очищает папку для сохранения изображений
process_folder(INPUT_PATH, OUTPUT_PATH)  #вызов функции рекурсивной обработки

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:21:30 2019

@author: Alisa Kuskova
"""
# Библиотеки для работы бота
import telebot
import logging
from telebot import apihelper

# Библиотеки для работы с файлами
from os.path import join

# Библиотека поиска места возникновения ошибок в программе, для отладки
import traceback

# Наша функция распознавания
from check_image import check_file


# Секретный ключ для подключения бота к telegram
TOKEN = 'INSERT SECRET TOKEN FROM TELEGRAM THERE'
USERNAME = 'Pokemonitor_Bot'

# Папка для скачивания изображений
IMAGE_FOLDER = './bot_downloaded_images'

# 

# Параметры использования прокси из-за блокировки Telegram
apihelper.proxy = {'https':'https://proxy_user:proxy_password@proxy_ip:proxy_port'}

# Создание бота
bot = telebot.TeleBot(TOKEN)

# Запуск логирования событий
logger = telebot.logger
telebot.logger.setLevel(logging.INFO)

# Регистрация обработчика команд start и help
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    logger.info(message)
    answer = "Здравствуйте! Я Pokemonitor_Bot, учебный проект по распознаванию покемонов"
    answer = answer + "\n В данный момент я способен распознавать следующих покемонов: Бульбазавр, Генгар, Джигглипафф, Дигглет, Дитто, Пикачу, Сквиртл, Тангела, Чармандер и Черизард"
    answer = answer + "\n Отправьте мне картинку с одним из этих покемонов и я попытаюсь угадать, кто это" 
    bot.send_message(message.chat.id, answer)
    
    
# Регистрация обработчика изображений, получаемых ботом
@bot.message_handler(content_types=['photo'])
def receive_image(message):
    logger.info(message)
    
    # Вежливо отвечаем, сообщая о получении изображения
    bot.reply_to(message, "спасибо, получил!")
    
    try:
        # Telegram для каждого изображения прикладывает копии разного размера.
        # Самое последнее содержит исходное изображение, нам оно и нужно
        # Получаем идентификатор последнего файла
        file_id = message.photo[-1].file_id
        
        # Получаем сведения о файле по идентификатору
        file_info = bot.get_file(file_id)
    
        # Получаем ссылку на скачивание файла
        downloaded_file = bot.download_file(file_info.file_path)
    
        # Сохраняем файл для диск для дальнейшей обработки
        filename = join(IMAGE_FOLDER, file_id+'.jpg')
        with open(filename, 'wb') as new_file:
            new_file.write(downloaded_file)
            
            result = check_file(filename)
            txt = 'Картинка похожа на '
            for r in result:
                txt = txt + r[0] + '(' + str(r[1]) + '%)'+"\n"
            bot.send_message(message.chat.id, txt)
    except:
        print("Неизвестная ошибка", traceback.format_exc())
    
# Запуск бота c параметрами: не останавливаться при ошибках, 
#    проверять новые сообщение каждые 1 сек, 
#    прерывать команду, если она занимает боль 20 сек
bot.polling(none_stop=True, interval=1, timeout=20) 
  
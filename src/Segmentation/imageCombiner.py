# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import PySimpleGUI as sg 
import os as os
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
import natsort
import sys 

#TODO ctrl+f and replace Window Title

def get_dir():
    layout = [[sg.Text("Enter the folder you wish to process")],
              [sg.Input(key='folder_location'), sg.FolderBrowse('Browse')],
              [sg.Button('Ok'), sg.Button('Cancel')]]

    window = sg.Window('Subimage Combiner', layout)
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Cancel':
            window.close()
            return None
        if event == 'Ok':
            folder_location = values['folder_location']
            if check_dir(folder_location):
                window.close()
                return folder_location
            else:
                sg.popup_ok('Not a valid folder.', text_color='red')

def get_size():
    layout = [[sg.Text("Enter the size of the orginal image")],
              [sg.Input(key='size_x', size=(11,1)), sg.Text('x'), sg.Input(key='size_y', size=(11,1))],
              [sg.Button('Ok'), sg.Button('Cancel')]]

    window = sg.Window('Subimage Combiner', layout)
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Cancel':
            window.close()
            return None
        if event == 'Ok':
            size_x = values["size_x"]
            size_y = values["size_y"]
            if(enforce_size(size_x) and enforce_size(size_y)):
                window.close()
                return (int(size_x), int(size_y))
            else:
                sg.popup_ok('Not a valid size', text_color='red')


def get_subimage_size():
    layout = [[sg.Text("Enter the size of the subimage (should be 500x500)")],
              [sg.Input(500, key='subimage_size_x', size=(11,1)), sg.Text('x'), sg.Input(500, key='subimage_size_y', size=(11,1))],
              [sg.Button('Ok'), sg.Button('Cancel')]]

    window = sg.Window('Subimage Combiner', layout)
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Cancel':
            window.close()
            return None
        if event == 'Ok':
            subimage_size_x = values["subimage_size_x"]
            subimage_size_y = values["subimage_size_y"]
            if(enforce_size(subimage_size_x) and enforce_size(subimage_size_y)):
                window.close()
                return (int(subimage_size_x), int(subimage_size_y))
            else:
                sg.popup_ok('Not a valid size', text_color='red')


def combine_images(dir, size_x, size_y, subimage_size_x, subimage_size_y):
    os.chdir(dir)
    print(dir)
    files = [f for f in listdir(dir) if isfile(join(dir, f)) ]
    files = natsort.natsorted(files)
    print(files)
    new_im = Image.new('RGB', (size_x,size_y))
    index = 0
    for i in range(0,size_y,subimage_size_y):
        for j in range(0,size_x,subimage_size_x):
            im = Image.open(files[index])
            im.thumbnail((subimage_size_x,subimage_size_y))
            new_im.paste(im, (j,i))
            index += 1
    return new_im
    pass


def save_image(new_im):
    """
    imarray should be a img
    """
    layout = [[sg.Text("Choose name and location to save your file")],
              [sg.Input(key='file_name'), sg.FileSaveAs('Browse')],
              [sg.Button('Save'), sg.Button('Cancel')]]

    window = sg.Window('Subimage Combiner', layout)
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Cancel':
            window.close()
            return None
        if event == 'Save':
            file_name = values['file_name']
            new_im.save(file_name)
            window.close()
            return None
            

def start():
    location = get_dir()
    if not location:
        return
    size = get_size()
    subimage_size = get_subimage_size()
    if size == None:
        return
    size_x, size_y = size
    subimage_size_x, subimage_size_y = subimage_size
    save_image(combine_images(location, size_x, size_y, subimage_size_x, subimage_size_y))
    sys.exit()

def check_dir(path):
    if not path:
        return False
    try:
        result = os.path.isdir(path)
    except:
        result = False
    return result

def enforce_size(num):
    num = num.strip()
    try:
        num = float(num)
    except:
        return False
    if num < 1:
        return False
    if num != int(num):
        return False
    return True

start()
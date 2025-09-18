import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob

def readClassificationSplitFolder(input_folder):
    '''
        Read data for classification problem (split folder by classes) into
            - input list: store image paths
            - label list: store answer classes
    '''
    # -> Glob all folders (classes)
    class_folder = glob.glob(input_folder + '*')

    input_list = []
    label_list = []
    
    for folder in class_folder:
        # -> input list
        input_files = glob.glob(folder + '/' + '*')
        input_list = input_list + input_files
        # -> label list
        label = folder.split('\\')[-1]
        label_texts = [label] * len(input_files)
        label_list = label_list + label_texts

    return input_list, label_list
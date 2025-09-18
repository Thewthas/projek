# <******************* Import *******************>

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# <******************* Internal Import *******************>

from DIPlib import *
from DIPlib.filters.frequency import *
from DIPlib.fourier import *
from skimage.exposure import equalize_hist
from DIPlib.segmentations import *
import skimage.morphology as skmorph
from DIPlib.morphology import *
from DIPlib.features.regions import *

# <******************* Main Script *******************>

DATABASE_PATH = "input/Leaves/"

if __name__ == "__main__":

    input_file1 = glob(DATABASE_PATH + "1/" + "*")
    input_file2 = glob(DATABASE_PATH + "2/" + "*")
    input_files = input_file1 + input_file2
    print(input_files)

    for f in input_files:

        input_img = cv.imread(f)
        rgb_img = cv.cvtColor(input_img,cv.COLOR_BGR2RGB)
        # gray_img = cv.cvtColor(input_img,cv.COLOR_BGR2GRAY)
        
        # _, seg_img = cv.threshold(gray_img,150,255,cv.THRESH_BINARY)
        # seg_img = colorRange(rgb_img,(104,154,41),r_cutoff=70)
        # center_color = (31,55,76)
        # upper_color = (27,0,75)
        # lower_color = (35,255,255)
        # seg_img = cv.inRange(rgb_img,lower_color,upper_color)
        gb_diff = rgb_img[:,:,1].astype(float) - rgb_img[:,:,2].astype(float)
        gb_diff = np.clip(gb_diff,0,255).astype(np.uint8)
        _,seg_img = cv.threshold(gb_diff,None,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        stre = skmorph.disk(9)

        # # morph_img = cv.erode(seg_img,stre)
        # morph_img = cv.morphologyEx(seg_img,cv.MORPH_OPEN,stre)
        morph_img = removeFragments(seg_img,thresh_ratio=0.05)
        morph_img = fillHoles(morph_img)
        morph_img = cv.morphologyEx(morph_img,cv.MORPH_CLOSE,stre)
        morph_img = fillHoles(morph_img)

        _,eccen =  regionBasedFeatures(morph_img,"eccentricity")
        print(eccen[0])
        
        if eccen[0] < 0.8:
            leaf_class = "1"
        else:
            leaf_class = "2"

        plt.subplot(1,2,1)
        plt.imshow(rgb_img)
        plt.subplot(1,2,2)
        plt.title(f"Leaf Class: {leaf_class}")
        plt.imshow(morph_img, cmap="gray")
        plt.show()
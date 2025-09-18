import cv2 as cv
import numpy as np
from DIPlib.filters.smoothing import gaussianFilter

def unsharpMasking(input_img, blurfilter_size, k):
    '''
        Unsharp Masking by edge finding from Gaussian Filtering
    '''
    # - Convert to float, operate with a negative number
    input_img = input_img.astype(float)

    ### -> Gaussian Filtering, Low components
    gauss_filter = gaussianFilter(blurfilter_size)
    low_img = cv.filter2D(input_img, -1, gauss_filter)

    ### -> Unsharp Masking, High components
    high_img = input_img -low_img

    ### -> High Boosting
    output_img = input_img + (k * high_img)

    ### -> Clip range into [0, 255]
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)

    return output_img

import cv2 as cv
import numpy as np

def removeFragments(input_img, thresh_ratio=0.01):
    '''
        Remove all object which are considered (by size) as fragments
    '''
    # -> Connected Components
    input_img = input_img.astype(np.uint8)
    _, label_img = cv.connectedComponents(input_img)

    # -> Count Number of pixel of each label
    labels, counts = np.unique(label_img, return_counts=True)

    # - Cut-off Value
    count_total = input_img.shape[0] * input_img.shape[1]
    count_thresh = int(count_total * thresh_ratio)

    # - Thresholding
    pass_index = np.argwhere(counts > count_thresh).flatten()
    assert len(pass_index) > 1, "All Objects are Fragments, Try reducing 'thresh_ratio' value"

    # - Pass Label/Group (EXCEPT - Label/Group 0)
    output_img = np.isin(label_img, pass_index[1:])
    output_img = output_img.astype(np.uint8)

    return output_img
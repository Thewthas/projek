import cv2 as cv
import numpy as np
from DIPlib.filters.edge import laplacianFilter

def lpcSharpening(input_img, lpc_center="negative", lpc_neighbors=4):
    '''
        Sharpening by Laplacian Edge Filering
    '''
    # - Convert to float, operate with a negative number
    input_img = input_img.astype(float)

    ### -> Laplacian Filtering
    lpc_filter = laplacianFilter(lpc_center, lpc_neighbors)
    edge_img = cv.filter2D(input_img, -1, lpc_filter)

    ### -> Sharpening by Adding "edge_img"
    output_img = input_img + ((-1) * edge_img)

    ### -> Clip range into [0, 255]
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)

    return output_img
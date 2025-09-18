import cv2 as cv

def gaussianFilter(filter_size):
    '''
       Gaussian Filter
    '''
    # -> Create Gaussian Filter
    gauss_filter = cv.getGaussianKernel(filter_size, -1)
    gauss_filter = gauss_filter * gauss_filter.T

    return gauss_filter
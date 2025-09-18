import numpy as np
from DIPlib.general import distanceMap

def idealFunction(distance_map, band_center, band_width):
    '''
        Ideal Function
    '''
    ideal_func = distance_map.copy()
    ideal_func = np.where((distance_map >= (band_center-band_width/2)) &
                          (distance_map <= (band_center+band_width/2)), 
                           1, 0)

    return ideal_func

def gaussianFunction(distance_map, band_center, band_width):
    '''
        Gaussian Function
    '''
    gauss_func = np.exp(-((distance_map**2-band_center**2) / (distance_map*band_width))**2)

    return gauss_func

def butterworthFunction(distance_map, band_center, band_width, n_order):
    '''
        Butterworth Function
    '''
    bw_func = 1 / (1 + ((distance_map**2-band_center**2) / (distance_map*band_width))**(2*n_order))

    return bw_func

def bandpassFilter(filter_size, band_center, band_width, filter_pos=None, filter_func="Gaussian", n_order=2):
    '''
        Band-pass Filter in Frequency Domain
    '''
    # -> Bandpass Filter defaultly locate at the center
    if filter_pos == None:
        filter_pos = (filter_size[0]//2, filter_size[1]//2)

    # -> Distance Map 2D from given position 'filter_pos'
    distance_map = distanceMap(filter_size, filter_pos)

    # -> Create Frequency Filter from selected Function
    filterFunction = {
                        "Ideal": idealFunction(distance_map, band_center, band_width),
                        "Gaussian": gaussianFunction(distance_map, band_center, band_width),
                        "Butterworth": butterworthFunction(distance_map, band_center, band_width, n_order)
                     }
    bp_filer = filterFunction[filter_func]

    return bp_filer
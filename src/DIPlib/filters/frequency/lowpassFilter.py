import numpy as np
from DIPlib.general import distanceMap

def idealFunction(distance_map, freq_cutoff):
    '''
        Ideal Function
    '''
    ideal_func = distance_map.copy()
    ideal_func[distance_map <= freq_cutoff] = 1
    ideal_func[distance_map > freq_cutoff] = 0

    return ideal_func

def gaussianFunction(distance_map, freq_cutoff):
    '''
        Gaussian Function
    '''
    gauss_func = np.exp(-distance_map**2 / (2*freq_cutoff**2))

    return gauss_func

def butterworthFunction(distance_map, freq_cutoff, n_order):
    '''
        Butterworth Function
    '''
    bw_func = 1 / (1 + (distance_map/freq_cutoff)**(2*n_order))

    return bw_func

def lowpassFilter(filter_size, freq_cutoff, filter_pos=None, filter_func="Gaussian", n_order=2):
    '''
        Low-pass Filter in Frequency Domain
    '''
    # -> Lowpass Filter defaultly locate at the center
    if filter_pos == None:
        filter_pos = (filter_size[0]//2, filter_size[1]//2)
    
    # -> Distance Map 2D from given position 'filter_pos'
    distance_map = distanceMap(filter_size, filter_pos)

    # -> Create Frequency Filter from selected Function
    filterFunction = {
                        "Ideal": idealFunction(distance_map, freq_cutoff),
                        "Gaussian": gaussianFunction(distance_map, freq_cutoff),
                        "Butterworth": butterworthFunction(distance_map, freq_cutoff, n_order)
                     }
    lp_filer = filterFunction[filter_func]

    return lp_filer
    
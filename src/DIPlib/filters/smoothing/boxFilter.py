import numpy as np

def boxFilter(filter_size):
    '''
        Box/Average Filter
    '''
    # -> Create Box/Average Filter
    box_filter = np.ones((filter_size, filter_size))
    box_filter = (1 / (filter_size * filter_size)) * box_filter

    return box_filter
import numpy as np
from DIPlib.filters.frequency import highpassFilter

def selectiveFilter(filter_size, select_pos, select_radius, pass_filter=True, func_type="Gaussian", n_order=2):
    '''
        Selective Position Filter in Frequency Domain
    '''
    # -> Filter Height & Width
    filter_height, filter_width = filter_size

    # -> Empty (Ban) Filter
    select_filter = np.ones((filter_height, filter_width))

    for pos, r in zip(select_pos, select_radius):
        # -> Create Selective Position (Ban) Filter
        pos = tuple(pos)
        pos_filter = highpassFilter(filter_size, r, pos, func_type, n_order)
        # -> Mirror Position, Mirror Selective Position (Ban) Filter
        mpos = (filter_height-pos[0], filter_width-pos[1])
        mpos_filter = highpassFilter(filter_size, r, mpos, func_type, n_order)
        # -> Merge (Ban) Filter
        select_filter = select_filter * pos_filter * mpos_filter

    # -> Selective (Pass or Ban) Filter
    if pass_filter:
        select_filter = 1 - select_filter

    return select_filter
from DIPlib.filters.frequency import lowpassFilter

def highpassFilter(filter_size, freq_cutoff, filter_pos=None, filter_func="Gaussian", n_order=2):
    '''
        High-pass Filter in Frequency Domain
    '''
    return 1 - lowpassFilter(filter_size, freq_cutoff, filter_pos, filter_func, n_order)
    
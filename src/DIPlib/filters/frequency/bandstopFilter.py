from DIPlib.filters.frequency import bandpassFilter

def bandstopFilter(filter_size, band_center, band_width, filter_pos=None, filter_func="Gaussian", n_order=2):
    '''
        Band-stop Filter in Frequency Domain
    '''
    return 1 - bandpassFilter(filter_size, band_center, band_width, filter_pos, filter_func, n_order)
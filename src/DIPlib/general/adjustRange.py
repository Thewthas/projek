import numpy as np

def adjustRange(input_array, input_range, output_range):
    '''
        Convert any value range array into specific range value array
        - input_range:    [input_min, input_max]
        - output_range:   [output_min, output_max]
    '''
    # -> Convert into [0, 1]
    norm_array = (input_array - input_range[0]) / (input_range[1] - input_range[0])
    # -> Convert [0, 1] into [min_val, max_val]
    output_array = (norm_array * (output_range[1] - output_range[0])) + output_range[0]
    
    return output_array
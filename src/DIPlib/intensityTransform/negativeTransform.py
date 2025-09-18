import numpy as np

def negativeTransform(input_array):
    '''
        Inverse the value of the "input_array"
    '''
    # -> Find maximum value of "input_array" datatype
    levels = np.iinfo(input_array.dtype).max - np.iinfo(input_array.dtype).min
    # -> Negative Transform
    output_array = levels - input_array

    return output_array
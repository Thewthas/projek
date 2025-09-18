import numpy as np

def logTransform(input_array, c=None, to_uint8=True):
    '''
        Scale "input_array" intensity by Logarithm
    '''
    # -> Default C
    if c == None:
        try:
            max_level = np.iinfo(input_array.dtype).max - np.iinfo(input_array.dtype).min
            # -> Convert "input_array", prevent overflow
            input_array = input_array.astype(float)
            c = max_level / (np.log(1 + np.max(input_array)))
        except:
            max_level = np.finfo(input_array.dtype).max - np.finfo(input_array.dtype).min
            c = max_level / (np.log(1 + np.max(input_array)))

    # -> Log Transform
    trans_array = c * np.log(1 + input_array)
    # -> Convert into default datatype
    if to_uint8:
        output_array = trans_array.astype(np.uint8)
    else:
        output_array = trans_array

    return output_array
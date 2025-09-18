import numpy as np
from DIPlib.general import adjustRange

def powerLawsTransform(input_array, gamma, c=1):
    '''
        Scale "input_array" intensity by Exponent
    '''
    # -> Convert to range [0, 1]
    norm_array = adjustRange(input_array, (0, 255), (0, 1))
    # -> Power Laws Transform
    trans_array = c * norm_array ** gamma
    # -> Convert to range [0, 256)
    output_array = adjustRange(trans_array, (0, 1), (0, 255))
    output_array = output_array.astype(np.uint8)

    return output_array
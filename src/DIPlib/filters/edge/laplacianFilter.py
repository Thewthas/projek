import numpy as np

def laplacianFilter(center="negative", neighbors=4):
    '''
        Laplacian Filter 3x3
    '''
    ### -> Laplacian Filter
    if neighbors == 4:
        lpc_filter = np.array([[0, 1, 0],
                               [1,-4, 1],
                               [0, 1, 0]])
        
    elif neighbors == 8:
        lpc_filter = np.array([[1, 1, 1],
                               [1,-8, 1],
                               [1, 1, 1]])
        
    ### -> Center Type
    if center == "positive":
        lpc_filter = lpc_filter * (-1)

    return lpc_filter
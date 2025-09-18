import numpy as np

def distanceMap(image_shape, ref_pos):
    '''
        Distance Map 2D for Function index
    '''
    map_height = image_shape[0]    # - Image Height
    map_width = image_shape[1]     # - Image Width
    
    # -> Preset Position Index
    v = np.arange(0.5*((map_height+1)%2), 0.5*((map_height+1)%2)+map_height)
    u = np.arange(0.5*((map_width+1)%2), 0.5*((map_width+1)%2)+map_width)
    uv, vv = np.meshgrid(u, v)

    # -> Distance Map 
    distance_map = ((uv-ref_pos[1])**2 + (vv-ref_pos[0])**2)**0.5

    return distance_map
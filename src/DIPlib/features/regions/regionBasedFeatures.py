import cv2 as cv
from skimage.measure import regionprops

def regionBasedFeatures(input_img, feature, intensity_img=None):
    '''
        Region-Based Feature Extraction
    '''
    _, label_img = cv.connectedComponents(input_img)
    object_list = regionprops(label_img, intensity_image=intensity_img)
    
    obj_img_list = []
    feature_list = []
    
    for object in object_list:
        # - Get cropped object image
        obj_img = object.image
        obj_img_list.append(obj_img)
        
        # - Get region-based feature value
        feature_val = getattr(object, feature, None)
        feature_list.append(feature_val)
        
    return obj_img_list, feature_list
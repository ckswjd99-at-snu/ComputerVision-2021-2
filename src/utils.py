import numpy as np
import cv2

def pad_black(image):
    padded = np.pad(image, 1, mode='constant', constant_values=0)
    return padded

def show_image(image):
    cv2.imshow('.', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def make_mask(image, contour):
    blank = np.zeros(image.shape)
    blank = cv2.fillPoly(image, contour, 1)
    return blank

def enlarge_black(image, size):
    newImage = image.copy()
    for i in range(size, len(image)):
        for j in range(size, len(image[i])):
            if image[i, j] == 0:
                newImage[i-size:i+size, j-size:j+size] = image[i-size:i+size, j-size:j+size] * 0
    return newImage

def hierarchy_height(hierarchy, index):
    if hierarchy[index][2] == -1:
        return 0
    
    children = []
    for i in range(len(hierarchy)):
        if hierarchy[i][3] == index:
            children.append(i)
    
    max_height = -1
    for c in children:
        max_height = max(max_height, hierarchy_height(hierarchy, c))
    
    return max_height + 1
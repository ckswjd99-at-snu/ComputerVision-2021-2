import cv2
import os
import numpy
import glob

import utils

def find_balloon_contour(image):
    image = image.copy()
    rgbImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # blurred_image = cv2.GaussianBlur(image, (3, 3), 3)

    # padded_image = utils.pad_black(image)

    thres = 200
    _, thres_image_inv = cv2.threshold(image, thres, 255, cv2.THRESH_BINARY_INV)
    thres_image = cv2.bitwise_not(thres_image_inv)

    enlarged_image = utils.enlarge_black(thres_image, 2)

    contours, [hierarchy] = cv2.findContours(enlarged_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy_heights = []
    for i in range(len(contours)):
        hierarchy_heights.append(utils.hierarchy_height(hierarchy, i))

    

    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        minWidth = 8
        minHeight = 8
        minHier = 1
        maxHier = 2
        if minWidth >= w or minHeight >= h:
            continue
        if hierarchy_heights[i] < minHier or maxHier < hierarchy_heights[i]:
            continue
        rgbImage = cv2.drawContours(rgbImage, [contours[i]], 0, (255, 0, 0), 2)

    
        
    utils.show_image(rgbImage)
    
    
    

# main run
i = 0
image_dir = '../test_images'
for image_path in glob.glob(image_dir+'/*.PNG'):
    image_orig = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
    find_balloon_contour(image_gray)

    print(i)
    i += 1
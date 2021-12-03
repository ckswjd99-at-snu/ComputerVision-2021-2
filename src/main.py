import cv2
import os
import numpy as np
import glob

import utils
import tempdata

def find_balloon_contour(image):
    image = image.copy()
    rgbImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # blurred_image = cv2.GaussianBlur(image, (3, 3), 3)

    # padded_image = utils.pad_black(image)

    thres = 200
    _, thres_image_inv = cv2.threshold(image, thres, 255, cv2.THRESH_BINARY_INV)
    thres_image = cv2.bitwise_not(thres_image_inv)

    enlarged_image = utils.enlarge_black(thres_image, 2)

    # reduced_image = utils.reduce_frag(enlarged_image, 5, 5, 10)
    # reduced_image = utils.reduce_treelike(reduced_image, 0.7)

    # utils.show_image(reduced_image)

    contours, [hierarchy] = cv2.findContours(enlarged_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy_heights = []
    for i in range(len(contours)):
        hierarchy_heights.append(utils.hierarchy_height(hierarchy, i))

    cnt_frag_mask = utils.contour_is_frag(contours, 5, 5, 10)
    cnt_treelike_mask = utils.contour_is_treelike(contours, 0.7)

    contours_to_use = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        minWidth = 8
        minHeight = 8
        minHier = 1
        maxHier = 3 # critical design parameter
        if minWidth >= w or minHeight >= h:
            continue
        if hierarchy_heights[i] < minHier or maxHier < hierarchy_heights[i]:
            continue
        if cnt_frag_mask[i] or cnt_treelike_mask[i]:
            continue
        rgbImage = cv2.drawContours(rgbImage, [contours[i]], 0, (255, 0, 0), 2)
        contours_to_use.append(contours[i])
        
    utils.show_image(rgbImage)

    return contours_to_use


def balloon_with_text(contours, bounding_box, image):
    contours_bounding_box = [[] for b in bounding_box]
    for i in range(len(bounding_box)):
        box = bounding_box[i]
        points = utils.nine_points(box)
        for cnt in contours:
            bounds = 9
            for p in points:
                if cv2.pointPolygonTest(cnt, p, False) < 0:
                    bounds -= 1
            if bounds > 6:
                contours_bounding_box[i].append(cnt)
    
    debug_image = image.copy()
    for contours in contours_bounding_box:
        cv2.drawContours(debug_image, contours, 0, (255, 0, 0))
    utils.show_image(debug_image)


# main run
i = 0
image_dir = '../test_images'
for image_path in glob.glob(image_dir+'/*.PNG'):
    image_orig = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
    
    balloon_candi_contours = find_balloon_contour(image_gray)
    mask = utils.make_masks(image_gray, balloon_candi_contours)
    masked_image = mask * image_gray
    utils.show_image(masked_image)

    temp_bounding_box = tempdata.bounding_box[image_path]

    image_debug = image_orig.copy()
    for box in temp_bounding_box:
        cv2.rectangle(image_debug, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (50, 255, 50), 2)
    utils.show_image(image_debug)

    balloon_candi_contours = balloon_with_text(balloon_candi_contours, temp_bounding_box, image_orig)

    print(i)
    i += 1
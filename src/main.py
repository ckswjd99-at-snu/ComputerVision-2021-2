import cv2
import os
import numpy as np
import glob
import math
import json

import utils
import tempdata
import ocr
import translate

def find_balloon_contour(image):
    image = image.copy()
    rgbImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # blurred_image = cv2.GaussianBlur(image, (3, 3), 3)

    # padded_image = utils.pad_black(image)

    thres = 200
    _, thres_image_inv = cv2.threshold(image, thres, 255, cv2.THRESH_BINARY_INV)
    thres_image = cv2.bitwise_not(thres_image_inv)

    enlarged_image = utils.enlarge_black(thres_image, 1)    # design parameter

    # reduced_image = utils.reduce_frag(enlarged_image, 5, 5, 10)
    # reduced_image = utils.reduce_treelike(reduced_image, 0.7)

    # utils.show_image(enlarged_image)

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
        maxHier = 2 # critical design parameter
        if minWidth >= w or minHeight >= h:
            continue
        if hierarchy_heights[i] < minHier or maxHier < hierarchy_heights[i]:
            continue
        if cnt_frag_mask[i] or cnt_treelike_mask[i]:
            continue
        contours_to_use.append(contours[i])
    
    #debug
    contours_to_use = utils.reduce_overlap_contour(contours_to_use)
    for cnt in contours_to_use:
        rgbImage = cv2.drawContours(rgbImage, [cnt], 0, (255, 0, 0), 2)

        
    # utils.show_image(rgbImage)

    return contours_to_use


def balloon_with_text(image, contours, bounding_box):
    contours_bounding_box = [[] for b in bounding_box]
    for i in range(len(bounding_box)):
        box = bounding_box[i]
        for cnt in contours:
            if utils.box_inside_contour(cnt, box):
                contours_bounding_box[i].append(cnt)                
    
    smallest_contour_bounding_box = []
    for contours in contours_bounding_box:
        min_area = math.inf
        min_contour = None
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if min_area > area:
                min_area = area
                min_contour = contours[i]
        smallest_contour_bounding_box.append(min_contour)
    
    debug_image = image.copy()
    for cnt in smallest_contour_bounding_box:
        if cnt is None:
            continue
        debug_image = cv2.drawContours(debug_image, [cnt], 0, (255, 0, 0), 1)
    # utils.show_image(debug_image)
    
    return smallest_contour_bounding_box, debug_image

def contour_with_text(image, contours, text_data):
    print(len(contours))
    textfrag_in_contours = [[] for c in contours]
    boxes = ocr.bounding_boxes_from(text_data)
    for i in range(len(textfrag_in_contours)):
        for j in range(len(boxes)):
            if utils.box_inside_contour(contours[i], boxes[j], 6):
                textfrag_in_contours[i].append(text_data[j].description)
        print(textfrag_in_contours[i])
    
    text_in_contours = [''.join(textfrag_in_contours[i]) for i in range(len(textfrag_in_contours))]
    return text_in_contours

def erase_balloon(image, contours):
    result_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for cnt in contours:
        mask = utils.make_mask(result_image, cnt, 255)
        fill_color = 255 if cv2.mean(result_image, mask=mask)[0] > 127 else 0
        result_image = cv2.fillPoly(result_image, [cnt], fill_color)
    return result_image


# main run
i = 0
image_dir = '../test_images'
for image_path in glob.glob(image_dir+'/*.PNG'):
    print(i)
    i += 1
    print(image_path)

    image_orig = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
    
    balloon_candi_contours = find_balloon_contour(image_gray)
    mask = utils.make_masks(image_gray, balloon_candi_contours)
    masked_image = mask * image_gray
    utils.show_image(masked_image)
    cv2.imwrite('./temp/first_filtered/'+str(i)+".PNG", masked_image)

    text_data, para_data = ocr.gv_ocr('./temp/first_filtered/'+str(i)+".PNG")
    utils.write_object('./temp/ocr_result/text_'+str(i)+'.json', text_data)
    utils.write_object('./temp/ocr_result/para_'+str(i)+'.json', para_data)

    text_box, text_list = ocr.text_data_from(para_data)
    utils.write_object('./temp/text_to_use/texts_'+str(i)+'.json', [text_box, text_list])
    text_list = translate.translate_text(text_list)

    # text_bounding_box = ocr.bounding_boxes_from(text_data)

    image_with_balloon = image_orig.copy()
    for box in text_box:
        cv2.rectangle(image_with_balloon, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (50, 255, 50), 2)
    utils.show_image(image_with_balloon)

    contours_bounding_box, debug_image = balloon_with_text(image_orig, balloon_candi_contours, text_box)
    cv2.imwrite('./temp/after_ocr/'+str(i)+".PNG", debug_image)

    contours_bounding_box = utils.reduce_same_contour(contours_bounding_box)

    image_blank_balloon = erase_balloon(image_orig, contours_bounding_box)
    cv2.imwrite('./temp/balloon_erased/'+str(i)+".PNG", image_blank_balloon)

    image_written = utils.write_text(image_blank_balloon, contours_bounding_box, text_box, text_list, split_by_word=1)
    cv2.imwrite('./temp/text_written/'+str(i)+".PNG", image_written)
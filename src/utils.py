import numpy as np
import cv2

def pad_black(image):
    padded = np.pad(image, 1, mode='constant', constant_values=0)
    return padded

def show_image(image):
    cv2.imshow('.', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def make_mask(image, contour, high=1):
    blank = np.zeros(image.shape, np.uint8)
    blank = cv2.fillPoly(blank, [contour], high)
    return blank

def make_masks(image, contours, high=1):
    blank = np.zeros(image.shape, np.uint8)
    for cnt in contours:
        blank = cv2.fillPoly(blank, [cnt], high)
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

def reduce_frag(image, min_width, min_height, min_area):
    result_image = image.copy()
    contours, [hierarchy] = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if w > min_width or h > min_height or area > min_area:
            continue
        mean_color = 255 if cv2.mean(image, mask=make_mask(image, cnt, 255))[0] > 127 else 0
        result_image = cv2.fillPoly(result_image, [cnt], 255-mean_color)
    return result_image

def reduce_treelike(image, thres):
    result_image = image.copy()
    debug_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
    contours, [hierarchy] = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        convex_hull = cv2.convexHull(cnt)
        cnt_area = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(convex_hull)

        treelike = cnt_area < hull_area * thres
        cv2.drawContours(debug_image, [cnt], 0, (255, 0, 0), 1)
        cv2.drawContours(debug_image, [convex_hull], 0, (50, 50, 255) if treelike else (50, 255, 50), 1)
        # cv2.putText(debug_image, str(cnt_area) + ", " + str(hull_area), convex_hull[0][0], 0, 0.0005*hull_area+0.2, (255, 0, 0), 1)

        if not treelike:
            continue
        mean_color = 255 if cv2.mean(image, mask=make_mask(image, cnt))[0] > 127 else 0
        result_image = cv2.fillPoly(result_image, [cnt], 255-mean_color)
        debug_image = cv2.fillPoly(debug_image, [cnt], (0, 0, 255))
    show_image(debug_image)
    return result_image

def contour_is_frag(contours, min_width, min_height, min_area):
    result = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if w > min_width or h > min_height or area > min_area:
            result.append(False)
        else:
            result.append(True)
        
    return result

def contour_is_treelike(contours, thres):
    result = []
    for cnt in contours:
        convex_hull = cv2.convexHull(cnt)
        cnt_area = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(convex_hull)

        treelike = cnt_area < hull_area * thres
        result.append(treelike)
        
    return result

def nine_points(box):
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    return [
        (x, y), (x+w/2, y), (x+w, y),
        (x, y+h/2), (x+w/2, y+h/2), (x+w, y+h/2),
        (x, y+h), (x+w/2, y+h), (x+w, y+h),
    ]
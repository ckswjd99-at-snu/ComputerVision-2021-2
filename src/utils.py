import numpy as np
import cv2
import pickle

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

def pickle_print(path):
    with open(path, 'rb') as fr:
        data = pickle.load(fr)
    print(data)

def contour_inside(outer, inner):
    if outer is None or inner is None:
        return False
    for point in inner:
        if cv2.pointPolygonTest(outer, (int(point[0][0]), int(point[0][1])), False) < 0:
            return False
    return True

def contour_same(cnt1, cnt2, error = 1):
    # length check
    if len(cnt1) != len(cnt2):
        return False
    
    # points check
    for i in range(len(cnt1)):
        p1 = np.array([cnt1[i][0][0], cnt1[i][0][1]])
        p2 = np.array([cnt2[i][0][0], cnt2[i][0][1]])
        if np.linalg.norm(p1 - p2) > error:
            return False
    
    return True


def reduce_overlap_contour(contours):
    contour_mask = [True for c in contours]
    for i in range(len(contours)):
        if contours[i] is None:
            contour_mask[i] = False
            continue
        for j in range(len(contours)):
            if i == j:
                continue
            if contours[j] is None:
                continue
            if contour_inside(contours[i], contours[j]):
                contour_mask[j] = False
    contour_alive = []
    for i in range(len(contours)):
        if contour_mask[i]:
            contour_alive.append(contours[i])

    return contour_alive

def reduce_same_contour(contours):
    mask = [True for c in contours]
    for i in range(len(contours)):
        if contours[i] is None:
            mask[i] = False
            continue
        for j in range(i+1, len(contours)):
            if mask[j] == False or contours[j] is None:
                mask[j] = False
                continue
            if contour_same(contours[i], contours[j]):
                mask[j] = False
    
    result = []
    for i in range(len(contours)):
        if mask[i]:
            result.append(contours[i])

    return result


def box_inside_contour(contour, box, thres = 6):
    containing = 0
    points = nine_points(box)
    for p in points:
        if contour is None:
            return False
        if cv2.pointPolygonTest(contour, p, False) >= 0:
            containing += 1
    return containing > thres

from PIL import ImageFont, ImageDraw, Image
def write_text(image, contours, box_list, text_list):
    result_image = image.copy()

    for i in range(len(box_list)):
        box = box_list[i]
        text = text_list[i]
        center = [int(box[0] + box[2]/2), int(box[1] + box[3]/2)]

        cv2.putText(result_image, text, center, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
    
    # show_image(result_image)


if __name__ == "__main__":
    pickle_print('./ocr_result.pickle')
from json import encoder
import numpy as np
import cv2
import pickle
import json

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
def get_textbox(text, width, height):
    size = 1

    font_dir = './fonts/meiryo.ttc'

    while True:
        # with size, render lines
        # print("trying with size ", size)
        font = ImageFont.truetype(font_dir, size)
        line_bag = []
        buffer = ''
        text_left = text
        
        while True:
            # add one symbol
            buffer += text_left[0]
            text_left = text_left[1:len(text_left)]
            line_size = font.getsize(buffer)

            # if line is full
            if line_size[0] > width:
                text_left = buffer[-1] + text_left
                buffer = buffer[0:len(buffer)-1]
                line_bag.append(buffer)
                buffer = ''
            
            # if all text is traversed
            if len(text_left) == 0:
                if len(buffer) > 0:
                    line_bag.append(buffer)
                    buffer = ''
                break
        
        lined_text = '\n'.join(line_bag)
        lined_size = font.getsize(lined_text)
        # print("now size ", lined_size[0], ", ", lined_size[1] * len(line_bag))
        # debug_image = Image.new('RGB', (width, height))
        # debug_drawer = ImageDraw.Draw(debug_image)
        # debug_drawer.text((0,0), lined_text, (255, 255, 255), font)
        # debug_image.show()

        if (lined_size[1]+4) * len(line_bag) > height:
            size -= 1
            break
        else:
            size += 1
            
    return lined_text, size

def make_box(contour):
    x, y, w, h = cv2.boundingRect(contour)
    bounding_box = [x + w/4, y + h/4, w/2, h/2]
    bounding_box = [int(elem) for elem in bounding_box]

    return bounding_box

def write_text(image, contours, box_list, text_list):
    result_image = Image.fromarray(image)
    font_dir = './fonts/meiryo.ttc'
    drawer = ImageDraw.Draw(result_image)

    for i in range(len(box_list)):
        box = box_list[i]
        text = text_list[i]
        [x, y, w, h] = box
        lined_text, size = get_textbox(text, w, h)

        font = ImageFont.truetype(font_dir, size)

        drawer.text((x, y), lined_text, 0, font)
    
    result_image = np.array(result_image, np.uint8)
    show_image(result_image)
    
    return result_image

def write_object(path, object):
    file = open(path, 'w', encoding='utf-8')
    td_str = json.dumps(object, ensure_ascii=False)
    file.write(td_str)
    file.close()


if __name__ == "__main__":
    get_textbox('hello this is for test man', 50, 50)
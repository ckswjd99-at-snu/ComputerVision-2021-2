import os
import io
import pickle
import math

from google.cloud import vision
from google.cloud.vision_v1 import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./angelic-cat-333914-c80aeb57e21a.json"

def gv_ocr(image_path):
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = types.Image(content=content)
    response = client.document_text_detection(image=image)
    texts = response.text_annotations
    full_text = response.full_text_annotation

    return texts, full_text

def bounding_boxes_from(text_data):
    boxes = []
    for data in text_data:
        vertices = data.bounding_poly.vertices
        min_x = math.inf
        max_x = -math.inf
        min_y = math.inf
        max_y = -math.inf
        for p in vertices:
            min_x = min(min_x, p.x)
            max_x = max(max_x, p.x)
            min_y = min(min_y, p.y)
            max_y = max(max_y, p.y)
        x = min_x
        y = min_y
        w = max_x - min_x
        h = max_y - min_y
        box = [x, y, w, h]
        boxes.append(box)
    
    return boxes

def vertices_to_box(vertices):
    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf
    for v in vertices:
        min_x = min(min_x, v.x)
        max_x = max(max_x, v.x)
        min_y = min(min_y, v.y)
        max_y = max(max_y, v.y)
    return [min_x, min_y, max_x-min_x, max_y-min_y]

def text_data_from(para_data):
    conf_thres = 0.8

    blocks = para_data.pages[0].blocks

    box_array = []
    text_array = []
    for block in blocks:
        paragraphs = block.paragraphs
        for para in paragraphs:
            para_box = vertices_to_box(para.bounding_box.vertices)
            if para.confidence < conf_thres:
                continue
            text = ''.join([''.join([symbol.text for symbol in word.symbols]) for word in para.words])

            box_array.append(para_box)
            text_array.append(text)
    
    return box_array, text_array





# if __name__ == "__main__":
#     text_data, paragraph_data = gv_ocr('./temp/first_filtered/0.PNG')
#     parsed_data = []
#     for data in text_data:
#         parsed_data.append({
#             'locale': "ja",
#             'description': "{}".format(data.description),
#             'bounding_poly': [(vertice.x, vertice.y) for vertice in data.bounding_poly.vertices]
#         })
#     parsed_para_data = []
#     for data in paragraph_data.pages[0]:
#         parsed_para_data.append({

#         })

#     with open("./ocr_result.pickle", 'wb') as fw:
#         pickle.dump(parsed_data, fw)
    
#     with open("./ocr_paragraph.pickle", 'wb') as fw:
#         pickle.dump(paragraph_data, fw)
    
if __name__ == "__main__":
    _, data = gv_ocr('./temp/first_filtered/0.PNG')
    
    boxes, texts = text_data_from(data)
    print(boxes)
    print(texts)
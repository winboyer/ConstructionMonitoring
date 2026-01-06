from paddleocr import PaddleOCR
import time
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
from utils.image_process import get_image_dimensions
from PIL import Image

ocr_model = PaddleOCR(
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# data_path = '/home/jinyfeng/datas/data_test/'
data_path = '/Users/jinyfeng/个人文档/zhongjian_works/AI课题/施工进度估计/隧道施工项目/data_test/'
start_time = time.time() 
# input_path = os.path.join(data_path, '视频识别/管片识别/微信图片_20251106160954_213_226.jpg')
input_path = os.path.join(data_path, '微信图片_20251216111447_251_2632.jpg')
image_width, image_height = get_image_dimensions(input_path)

max_dimension = 1280 * 1280 # max 1296x1296 pixels
if image_width * image_height > max_dimension:
    print(f"Original image size: {image_width}x{image_height}, resizing for OCR model")
    scale_factor = (max_dimension / (image_width * image_height)) ** 0.5
    new_width = int(image_width * scale_factor)
    new_height = int(image_height * scale_factor)
    print(f"Resized image size: {new_width}x{new_height}")

    img = Image.open(input_path)
    img = img.resize((new_width, new_height))
    result = ocr_model.predict(input=img)
else:
    result = ocr_model.predict(input=input_path)

print(len(result), result)
angle = result[0].get('angle', 0)
print(f"Detected angle: {angle}")


if angle == 90 or angle == 270:
    temp = image_width
    image_width = image_height
    image_height = temp


dt_polys = result[0].get('dt_polys', [])
# print(len(dt_polys), dt_polys)

rec_polys = result[0].get('rec_polys', [])
# print(len(rec_polys), rec_polys[0])
# print(rec_polys[0].shape)

rec_texts = result[0].get('rec_texts', [])
print(len(rec_texts),rec_texts)

project_name = ""
product_name_num = 0
product_name_list = []
product_name_coords = []
ring_id_list = []
ring_id_coords = []
product_cnt = 0
product_cnt_list = []
for idx, text in enumerate(rec_texts):
    print("idx:", idx, "text:", text)
    if "轨道交通7号线" in text:
        project_name = text
        print("施工项目：", text)
        print(idx, rec_polys[idx])
    elif "型" in text and "管片" in text:
        temp = text[:2]+'-'+text[-4:]
        print('产品名称1', temp)
        print(idx, rec_polys[idx])
        product_name_list.append(temp)
        center_x = (rec_polys[idx][0][0] + rec_polys[idx][1][0]+rec_polys[idx][2][0] + rec_polys[idx][3][0]) / 2
        center_y = (rec_polys[idx][0][1] + rec_polys[idx][1][1]+rec_polys[idx][2][1] + rec_polys[idx][3][1]) / 2
        product_name_coords.append((center_x, center_y))
    elif len(re.findall(r'-', text)) == 2:
        print('环号：', text)
        print(idx, rec_polys[idx])
        ring_id_list.append(text)
        center_x = (rec_polys[idx][0][0] + rec_polys[idx][1][0]+rec_polys[idx][2][0] + rec_polys[idx][3][0]) / 2
        center_y = (rec_polys[idx][0][1] + rec_polys[idx][1][1]+rec_polys[idx][2][1] + rec_polys[idx][3][1]) / 2
        ring_id_coords.append((center_x, center_y))
    elif bool(re.search(r'\.', text)) and bool(re.search(r'm', text)):
        print('产品名称2：', text)
        print(idx, rec_polys[idx])
        center_x = (rec_polys[idx][0][0] + rec_polys[idx][1][0]+rec_polys[idx][2][0] + rec_polys[idx][3][0]) / 2
        center_y = (rec_polys[idx][0][1] + rec_polys[idx][1][1]+rec_polys[idx][2][1] + rec_polys[idx][3][1]) / 2
        if center_x < image_width / 2:
            if not text[0]=="（" and not text[0]=="(":
                text = "(" + text
            if not text[-1]=="）" and not text[-1]==")":
                text = text + ")"
            product_name_list[product_name_num] = product_name_list[product_name_num] + text
            product_name_num += 1
    elif text.isdigit() and int(text) < 10:
        center_x = (rec_polys[idx][0][0] + rec_polys[idx][1][0]+rec_polys[idx][2][0] + rec_polys[idx][3][0]) / 2
        center_y = (rec_polys[idx][0][1] + rec_polys[idx][1][1]+rec_polys[idx][2][1] + rec_polys[idx][3][1]) / 2
        print('数量候选：', text)

        print(len(product_cnt_list), center_x)
        if len(product_cnt_list) < 1 and center_x > image_width / 2:
            print('product_cnt', product_cnt)
            product_cnt = int(text)
            print("数量：", text)
            product_cnt_list.append(product_cnt)
            
print(len(product_cnt_list), len(product_name_list))
if len(product_cnt_list) < len(product_name_list):
    product_cnt_list.extend([product_cnt] * (len(product_name_list) - len(product_cnt_list)))
print(len(product_cnt_list), len(product_name_list))

print("施工项目名称：", project_name)
print("产品名称列表：", product_name_list)
print("产品名称坐标列表：", product_name_coords)
print("环号列表：", ring_id_list)
print("环号坐标列表：", ring_id_coords)
print("产品数量：", product_cnt)
print("产品数量列表：", product_cnt_list)

end_time = time.time()
print(f"Inference time: {end_time - start_time:.2f} seconds")
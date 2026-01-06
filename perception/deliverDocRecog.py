from paddleocr import PaddleOCR
import time
import os
import re

ocr = PaddleOCR(
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True)

data_path = '/home/jinyfeng/datas/data_test/'
data_path = '/Users/jinyfeng/个人文档/zhongjian_works/AI课题/施工进度估计/隧道施工项目/data_test/'

start_time = time.time()
# input_path = os.path.join(data_path, '视频识别/管片识别/微信图片_20251106160954_213_226.jpg')
input_path = os.path.join(data_path, '微信图片_20251216111447_251_2632.jpg')
result = ocr.predict(input=input_path)

print(len(result))
angle = result[0].get('angle', 0)
print(f"Detected angle: {angle}")

dt_polys = result[0].get('dt_polys', [])
# print(len(dt_polys), dt_polys)

rec_polys = result[0].get('rec_polys', [])
# print(len(rec_polys), rec_polys[0])
# print(rec_polys[0].shape)

rec_texts = result[0].get('rec_texts', [])
print(len(rec_texts),rec_texts)

project_name = ""
product_name_list = []
product_name_coords = []
ring_number_list = []
ring_number_coords = []
product_cnt = 0
product_cnt_list = []
for idx, text in enumerate(rec_texts):
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
        ring_number_list.append(text)
        center_x = (rec_polys[idx][0][0] + rec_polys[idx][1][0]+rec_polys[idx][2][0] + rec_polys[idx][3][0]) / 2
        center_y = (rec_polys[idx][0][1] + rec_polys[idx][1][1]+rec_polys[idx][2][1] + rec_polys[idx][3][1]) / 2
        ring_number_coords.append((center_x, center_y))
    elif bool(re.search(r'\.', text)) and bool(re.search(r'm', text)):
        print('产品名称2：', text)
        print(idx, rec_polys[idx])
        center_x = (rec_polys[idx][0][0] + rec_polys[idx][1][0]+rec_polys[idx][2][0] + rec_polys[idx][3][0]) / 2
        center_y = (rec_polys[idx][0][1] + rec_polys[idx][1][1]+rec_polys[idx][2][1] + rec_polys[idx][3][1]) / 2
        if center_x < ring_number_coords[idx][0]:
            product_name_list[idx] = product_name_list[idx] + text
    elif text.isdigit() and int(text) < 10:
        center_x = (rec_polys[idx][0][0] + rec_polys[idx][1][0]+rec_polys[idx][2][0] + rec_polys[idx][3][0]) / 2
        center_y = (rec_polys[idx][0][1] + rec_polys[idx][1][1]+rec_polys[idx][2][1] + rec_polys[idx][3][1]) / 2
        if len(ring_number_list) > 0 and center_x > ring_number_coords[0][0]:
            if len(product_cnt_list) < 1:
                product_cnt = int(text)
            product_cnt_list.append(product_cnt)
            print("数量：", text)

            
print("施工项目名称：", project_name)
print("产品名称列表：", product_name_list)
print("产品名称坐标列表：", product_name_coords)
print("环号列表：", ring_number_list)
print("环号坐标列表：", ring_number_coords)
print("产品数量列表：", product_cnt_list)

end_time = time.time()
print(f"Inference time: {end_time - start_time:.2f} seconds")
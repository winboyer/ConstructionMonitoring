from paddleocr import PaddleOCR
import time
import os
from PIL import Image
import cv2
import json

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

data_path = '/home/jinyfeng/datas/data_test/'

# Run OCR inference on a sample image 
start_time = time.time()
input_path = os.path.join(data_path, '视频识别/渣土车识别/MVIMG_20251127_104611.jpg')
# 读取图片
img = Image.open(input_path)
# 根据像素点位置裁剪图片
cropped_img = img.crop((1000, 1700, 1600, 2100))
# 保存临时裁剪后的图片
cropped_path = os.path.join(data_path, 'cropped_temp.jpg')
cropped_img.save(cropped_path)
input_path = cropped_path
result = ocr.predict(input=input_path)

# 获取 rec_texts 字段并保存为 JSON 文件
rec_texts = result[0].get('rec_texts', [])
print(rec_texts)
end_time = time.time()
print(f"Inference time 11111111111111: {end_time - start_time:.2f} seconds")


start_time = time.time()
input_path = os.path.join(data_path, '视频识别/渣土车识别/MVIMG_20251127_104630.jpg')
# 读取图片
img = Image.open(input_path)
# 根据像素点位置裁剪图片
cropped_img = img.crop((800, 1800, 2000, 2500))
# 保存临时裁剪后的图片
cropped_path = os.path.join(data_path, 'cropped_temp.jpg')
cropped_img.save(cropped_path)
input_path = cropped_path
result = ocr.predict(input=input_path)
rec_texts = result[0].get('rec_texts', [])
print(rec_texts)
end_time = time.time()
print(f"Inference time 22222222222222: {end_time - start_time:.2f} seconds")


start_time = time.time()
input_path = os.path.join(data_path, '视频识别/渣土车识别/微信图片_2025-11-28_110120_403.jpg')
# 读取图片
img = Image.open(input_path)
# 根据像素点位置裁剪图片
cropped_img = img.crop((1700, 1400, 2150, 1680))
# 保存临时裁剪后的图片
cropped_path = os.path.join(data_path, 'cropped_temp.jpg')
cropped_img.save(cropped_path)
input_path = cropped_path
result = ocr.predict(input=input_path)
rec_texts = result[0].get('rec_texts', [])
print(rec_texts)
end_time = time.time()
print(f"Inference time 33333333333333333: {end_time - start_time:.2f} seconds")


start_time = time.time()
input_path = os.path.join(data_path, '视频识别/渣土车识别/微信视频2025-11-28_110908_704.mp4')
# 读取视频
video = cv2.VideoCapture(input_path)
video.set(cv2.CAP_PROP_POS_MSEC, 2000)  # 跳转到第2秒
success, frame = video.read()
if success:
    frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_img.save('temp_frame.jpg')
    cropped_img = frame_img.crop((0, 470, 580, 800))
    cropped_path = os.path.join(data_path, 'cropped_temp.jpg')
    cropped_img.save(cropped_path)
    input_path = cropped_path
else:
    raise RuntimeError("无法读取视频帧")
video.release()
result = ocr.predict(input=input_path)
rec_texts = result[0].get('rec_texts', [])
print(rec_texts)
end_time = time.time()
print(f"Inference time 44444444444444: {end_time - start_time:.2f} seconds")

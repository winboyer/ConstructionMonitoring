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
# start_time = time.time()
# filename = '视频识别/电瓶车识别/MVIMG_20251127_102819.jpg'          #编号为4
# filename = '视频识别/电瓶车识别/MVIMG_20251127_103146.jpg'          #编号为3
# input_path = os.path.join(data_path, filename)
# # 读取图片
# img = Image.open(input_path)
# # 根据像素点位置裁剪图片
# # cropped_img = img.crop((0, 750, 1900, 3250))
# cropped_img = img.crop((1000, 1020, 2100, 2600))
# # 保存临时裁剪后的图片
# cropped_path = os.path.join(data_path, 'cropped_temp.jpg')
# cropped_img.save(cropped_path)
# input_path = cropped_path
# result = ocr.predict(input=input_path)

# # 获取 rec_texts 字段并保存为 JSON 文件
# rec_texts = result[0].get('rec_texts', [])
# print(rec_texts)
# end_time = time.time()
# print(f"Inference time 11111111111111: {end_time - start_time:.2f} seconds")






start_time = time.time()
# filename = '视频识别/电瓶车识别/11d2a688edb57e2aa7ffccc025e1a31d.mp4' # 56s- 67s能看清电瓶车编号 1，暂时无法识别
# filename = '视频识别/电瓶车识别/65d32a3f5cb43d01f884948516c068e4.mp4' # 64s- 90s能看清电瓶车编号 2
filename = '视频识别/电瓶车识别/92110db24cf85770cd93354825b7e059.mp4' # 1s- 7s能看清电瓶车编号 1, 暂时无法识别

input_path = os.path.join(data_path, filename) 
# 读取视频
video = cv2.VideoCapture(input_path)
# video.set(cv2.CAP_PROP_POS_MSEC, 65000)  # 跳转到第X秒识别编号2
video.set(cv2.CAP_PROP_POS_MSEC, 7000)  # 跳转到第X秒
success, frame = video.read()
if success:
    frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    print(frame_img.size, frame_img.height, frame_img.width)
    frame_img.save('temp_frame.jpg')
    # cropped_img = frame_img.crop((360, 450, 690, frame_img.height if 730 > frame_img.height else 730))
    cropped_img = frame_img.crop((400, 320, 700, frame_img.height if 700 > frame_img.height else 700))
    cropped_path = 'temp_cropped.jpg'
    cropped_img.save(cropped_path)
    input_path = cropped_path
else:
    raise RuntimeError("无法读取视频帧")
video.release()
result = ocr.predict(input=input_path)
rec_texts = result[0].get('rec_texts', [])
print(rec_texts)
end_time = time.time()
print(f"Inference time : {end_time - start_time:.2f} seconds")

from paddleocr import PaddleOCR
import time
import os

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

data_path = '/home/jinyfeng/datas/data_test/'

start_time = time.time()
input_path = os.path.join(data_path, '视频识别/管片识别/微信图片_20251106160954_213_226.jpg')
result = ocr.predict(input=input_path)

print(len(result))
rec_texts = result[0].get('rec_texts', [])
print(len(rec_texts),rec_texts)

end_time = time.time()
print(f"Inference time: {end_time - start_time:.2f} seconds")
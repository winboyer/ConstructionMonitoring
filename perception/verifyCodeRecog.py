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


def verify_code(file_path):
    start_time = time.time()

    result = ocr.predict(input=file_path)
    rec_texts = result[0].get('rec_texts', [])
    print(rec_texts)
    end_time = time.time()
    print(f"Inference time : {end_time - start_time:.2f} seconds")




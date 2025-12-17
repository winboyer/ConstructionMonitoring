import json
import requests
import time
from PIL import Image
from io import BytesIO
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# Read data from file via API with parameters
read_url = 'http://117.132.13.99:9210/tpp/common/captcha/init'
try:
    response = requests.get(read_url, timeout=5)
    response.raise_for_status()
    print(f"Success: {response.status_code}")
    print(response.content)
    print(response.json())
    
    data = response.json()
    content_result = data.get("result")
    # print(json.dumps(data, ensure_ascii=False, indent=2))
    # content = data.get("message")
    print(f"File Content: {content_result}")
    
    new_url = "http://117.132.13.99:9210/tpp/common/captcha/draw/" + content_result
    print(new_url)
    response_new = requests.get(new_url, timeout=5)
    response_new.raise_for_status()
    print(f"Success: {response.status_code}")    
    # print(response_new.content)
    img = Image.open(BytesIO(response_new.content))
    filepath = './captcha.png'
    img.save(filepath)
    print("Image saved successfully")
    start_time = time.time()

    result = ocr.predict(input=filepath)
    rec_texts = result[0].get('rec_texts', [])
    print(rec_texts)
    end_time = time.time()
    print(f"Inference time : {end_time - start_time:.2f} seconds")

except requests.exceptions.RequestException as e:
    print(f"Read Error: {e}")

import json
import requests
import time
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import ddddocr
from flask import Flask, jsonify, request
import os


ocr_v1 = ddddocr.DdddOcr()
ocr_v2 = ddddocr.DdddOcr(beta=True)

image = open("captcha.png", "rb").read()
result = ocr_v1.classification(image)
result = ocr_v2.classification(image)
print(result)

app = Flask(__name__)

@app.route('/captcha/recognize', methods=['GET'])
def recognize_captcha():
    """Recognize captcha from the remote API and return the result"""
    try:
        # Get captcha ID from remote API
        read_url = 'http://117.132.13.99:9210/tpp/common/captcha/init'
        response = requests.get(read_url, timeout=5)
        response.raise_for_status()

        data = response.json()
        content_result = data.get("result")

        # Download captcha image
        new_url = f"http://117.132.13.99:9210/tpp/common/captcha/draw/{content_result}"
        response_new = requests.get(new_url, timeout=5)
        response_new.raise_for_status()
        
        img = Image.open(BytesIO(response_new.content))
        
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)

        # Use ddddocr v2 for recognition
        start_time = time.time()
        # result = ocr_v1.classification(img)
        result = ocr_v2.classification(img)
        end_time = time.time()
        print(f"recognition result :{result}, Inference time : {end_time - start_time:.2f} seconds")
        position = (10, 0) # (x, y) coordinates for the top-left corner of the text
        draw.text(position, result, fill=(0,255,0))

        # Save the captcha image with naming convention: result+content_result
        foldername = 'captcha_images'
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        img.save(f"{foldername}/{result}_{content_result}.png")
        draw_img.save(f"{foldername}/{result}_{content_result}_with_text.png")
        return jsonify({
            'status': response.status_code,
            'captcha_id': content_result,
            'captcha_code': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2512, debug=False)
from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
from io import BytesIO

from paddleocr import PaddleOCR
import time
import os

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

app = Flask(__name__)

@app.route('/recognize', methods=['POST'])
def recognize_delivery_document():
    """
    POST endpoint for delivery document recognition
    Request: JSON with base64 encoded image
    Response: JSON with recognized delivery document content
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Process image and extract delivery document info
        timestamp = int(time.time())
        foldername = 'deliverDocImgs'
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        img_save_path = f'{foldername}/image_{timestamp}.jpg'
        cv2.imwrite(img_save_path, img)
        result = extract_document_info(img_save_path)
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def extract_document_info(img_path):
    """Extract delivery document information from image"""
    # Placeholder for OCR/document recognition logic
    result = ocr.predict(input=img_path)
    print(len(result))
    rec_texts = result[0].get('rec_texts', [])
    print(len(rec_texts),rec_texts)
    
    return {
        'sender': '',
        'receiver': '',
        'address': '',
        'phone': '',
        'items': [],
        'total': 0
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
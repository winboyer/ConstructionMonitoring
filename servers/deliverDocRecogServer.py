from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
from io import BytesIO

from perception.paddleocrRecog import DocumentRecognizer 

import time
import os

app = Flask(__name__)

# Global recognizer instance (thread-safe)
recognizer = DocumentRecognizer()

@app.route('/deliverDoc/recognize', methods=['POST'])
def recognize_delivery_document():
    """
    POST endpoint for delivery document recognition
    Request: JSON with base64 encoded image
    Response: JSON with recognized delivery document content
    """
    try:
        data = request.get_json()
        print(f"Received request data ! ")

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
        result = recognizer.extract_info(img_save_path)
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=2601)
    app.run(debug=False, host='0.0.0.0', port=2601)
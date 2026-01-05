import re
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional
import cv2
print(cv2.__version__)
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perception.paddleocrRecog import DocumentRecognizer
import easyocr
from ultralytics import YOLO
from paddleocr import PaddleOCR

class OCRRecognizer:
    def __init__(self, model_path):
        """
        Initialize the security client.
        """
        # self.det_model = YOLO(model_path)
        # self.ocr_model = DocumentRecognizer()
        # self.easy_ocr_reader = easyocr.Reader(['ch_sim','en'], gpu=False)
        
    def detect_number(self, frame):
        """
        Detect safety gear in frame (placeholder for actual detection model).
        
        Returns:
            Tuple of (has_helmet, has_reflective_vest)
        """
                   

        # frame_copy = frame.copy()
        timestamp = None

        suffix = r"[A-Z0-9]{5}" #车牌号后面5位，可字母或数字（排除I/O）

        # result = self.det_model.predict(source=frame)
        # print(f"Detection Results: {results}")
        # print(len(results))
        # cls = results[0].boxes.cls.cpu().numpy()  
        # print(f"Classes: {cls}")    


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        ocr_info = self.ocr_model.extract_info(binary)
        print(f"OCR Info: {ocr_info}")
        # for ocr in ocr_info.get('result', []):
        #     print(f"OCR Result: {ocr}, length: {len(ocr)}")
            
        #     # Simple regex to match license plate patterns (customize as needed)
        #     # print(ocr[:2], ocr[-5:])
        #     # if re.match(plate_pattern_pre, ocr[:2]) and re.match(suffix, ocr[-5:]):
        #     #     truck_number = ocr[:2] + ocr[-5:]
        #     #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     #     cv2.imwrite(f"{image_save_folder}/image_{timestamp}.jpg", frame)
        #     #     break
        result = ocr_info.get('result', [])
        print(f"OCR Results: {result}")
        
        # result = self.easy_ocr_reader.readtext(frame)
        # print(f"EasyOCR Results: {result}") 

        return result, frame, timestamp

    def recognize_num_from_video(self, video_stream: str) -> Tuple[Optional[str], Optional[any], Optional[str]]:
        """
        
        Args:
        
        Returns:
        """
        try:            
            cap = cv2.VideoCapture(video_stream)
            if not cap.isOpened():
                print(f"Failed to open video source: {video_stream}")
                cap.release()
                return None, None, None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to read frame")
                    time.sleep(10)
                    continue
                recog_number, ret_frame, timestamp = self.detect_number(frame)
                if recog_number:
                    cap.release()
                    return recog_number, ret_frame, timestamp              
                time.sleep(30)      
        except Exception as e:
            cap.release()
            return False, f"Error: {str(e)}", None

if __name__ == "__main__":
    model_path = "/Users/jinyfeng/tools/object-det/yolo11n.pt"  # Update with your model directory
    
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    recognizer = OCRRecognizer(model_path)
    
    # file_path = "/Users/jinyfeng/个人文档/zhongjian_works/AI课题/施工进度估计/隧道施工项目/data_test/视频识别/2026-01-04_162450_357.mp4"
    file_path = "/Users/jinyfeng/Downloads/c29215b3-60cd-4003-8d04-583059998a6d.png"
    # file_path = "/Users/jinyfeng/Downloads/binary_image.jpg"
    # Determine if the file is an image or video
    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
        # Process as video
        recog_number, ret_frame, timestamp = recognizer.recognize_num_from_video(file_path)
        print(f"Recognized Number: {recog_number}")
        cv2.imshow("Result", ret_frame)
        cv2.waitKey(0)

    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        # Process as image
        image_path = file_path
    
        frame = cv2.imread(image_path)
        print(f"Original shape: {frame.shape}")
        # cv2.imshow("gray", frame)
        # cv2.waitKey(0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(f"Gray shape: {gray.shape}")
        gray_new = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        print(f"gray_new shape: {gray_new.shape}")
        # cv2.imwrite("gray_image.jpg", gray)
        # cv2.imshow("gray", gray)
        # cv2.waitKey(0)
        
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Binary shape: {binary.shape}")
        binary_new = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        print(f"binary_new shape: {binary_new.shape}")
        # cv2.imwrite("binary_image.jpg", binary)
        # cv2.imshow("bibinary_newnary", binary_new)
        # cv2.waitKey(0)

        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False)
        
        result = ocr.predict(input=binary_new)
        print(result, len(result))
        rec_texts = result[0].get('rec_texts', []) if result else []
        print(len(rec_texts), rec_texts)



        # truck_number, ret_frame, timestamp = recognizer.detect_truck(frame)
        # print(f"Results: {results}")
        # cv2.imshow("Result", ret_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


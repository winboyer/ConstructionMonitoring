import re
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional
import cv2
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perception.paddleocrRecog import DocumentRecognizer
from ultralytics import YOLO
import requests

class TruckRecognizer:
    def __init__(self, model_path):
        """
        Initialize the security client.
        """
        self.det_model = YOLO(model_path)
        self.ocr_model = DocumentRecognizer()
        
    def detect_truck(self, frame):
        """
        Detect safety gear in frame (placeholder for actual detection model).
        
        Returns:
            Tuple of (has_helmet, has_reflective_vest)
        """
        
        # results = self.det_model.predict(source=frame, conf=0.5)
        results = self.det_model.predict(source=frame)

        # print(f"result boxes: {result.boxes}")
        # print(f"result boxes: {result.boxes.conf, result.boxes.cls, result.boxes.xywh, result.boxes.xyxy}")
        xyxy = results[0].boxes.xyxy.cpu().numpy()
        xywh = results[0].boxes.xywh.cpu().numpy()
        conf = results[0].boxes.conf.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy()                

        class_names = self.det_model.names
        # frame_copy = frame.copy()
        truck_number = None
        timestamp = None

        province_abbr = r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼台使领]{1}" #省份简称汉字
        suffix = r"[A-Z0-9]{5}" #车牌号后面5位，可字母或数字（排除I/O）
        # plate_pattern = f"^{province_abbr}[A-Z]{1}{suffix}$"
        plate_pattern = province_abbr + r"[A-Z]{1}" + suffix
        plate_pattern_pre = province_abbr + r"[A-Z]{1}"

        image_save_folder = Path("./detected_trucks")
        if not image_save_folder.exists():
            image_save_folder.mkdir(parents=True, exist_ok=True)

        for i, c in enumerate(cls):
            # print(f"Class: {class_names[int(c)]}")
            if class_names[int(c)] == 'truck':
                print("Detected a truck!")
                # cv2.rectangle(frame_copy, (int(xyxy[i][0]), int(xyxy[i][1])), (int(xyxy[i][2]), int(xyxy[i][3])), (0, 255, 0), 2)
                # print(f"xyxy: {xyxy}, xywh: {xywh}, conf: {conf}, cls: {cls}")
                crop_img = frame[int(xyxy[i][1]):int(xyxy[i][3]), int(xyxy[i][0]):int(xyxy[i][2])]
                cv2.imshow("crop_img", crop_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                ocr_info = self.ocr_model.extract_info(crop_img)
                for ocr in ocr_info.get('result', []):
                    # print(f"OCR Result: {ocr}, length: {len(ocr)}")
                    if len(ocr) < 7:
                        continue
                    # Simple regex to match license plate patterns (customize as needed)
                    # print(ocr[:2], ocr[-5:])
                    if re.match(plate_pattern_pre, ocr[:2]) and re.match(suffix, ocr[-5:]):
                        truck_number = ocr[:2] + ocr[-5:]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f"{image_save_folder}/image_{timestamp}.jpg", frame)
                        break
        print(f"Detected truck number: {truck_number}")
        # ret_frame_b64 = bytes_to_base64(image_bytes)
        return truck_number, frame, timestamp

    def recognize_trucknum_from_rtsp(self, video_stream: str, interval=300) -> Tuple[Optional[str], Optional[any], Optional[str]]:
        """
        
        Args:
        
        Returns:
        """
        try:
            cap = None
            max_retries = 3
            for attempt in range(max_retries):
                cap = cv2.VideoCapture(video_stream)
                if cap.isOpened():
                    break
                print(f"Failed to open video source: {video_stream} (Attempt {attempt + 1}/{max_retries})")
                cap.release()
                time.sleep(2)  # Wait before retrying
            
            if cap is None or not cap.isOpened():
                print(f"Failed to open video source after {max_retries} attempts: {video_stream}")
                if cap is not None:
                    cap.release()
                return 
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to read truckRecog frame")
                    time.sleep(1)
                    continue
                truck_number, ret_frame, timestamp = self.detect_truck(frame)
                if truck_number:
                    image_bytes = cv2.imencode('.jpg', ret_frame)[1].tobytes()
                    response = requests.post(
                        "http://112.124.54.138:5001/api/loadcars/image",
                        json={
                            "licenSeplate": truck_number,
                            "timestamp": timestamp,
                            "imageFile": image_bytes
                        }
                    )
                    print(f"Posted truck plate info to server: {response.status_code}, {response.text}")
                    time.sleep(interval) 
                
                time.sleep(interval//2)      
        except Exception as e:
            cap.release()
            print(f"Error processing video stream: {video_stream}, Exception:{e}")

if __name__ == "__main__":
    model_path = "/Users/jinyfeng/tools/object-det/yolo11n.pt"  # Update with your model directory
    
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    recognizer = TruckRecognizer(model_path)
    
    # file_path = "/Users/jinyfeng/个人文档/zhongjian_works/AI课题/施工进度估计/隧道施工项目/data_test/视频识别/渣土车识别/MVIMG_20251127_104611.jpg"  # Update with your image path
    # file_path = "/Users/jinyfeng/个人文档/zhongjian_works/AI课题/施工进度估计/隧道施工项目/data_test/视频识别/渣土车识别/MVIMG_20251127_104630.jpg"  # Update with your image path
    file_path = "/Users/jinyfeng/个人文档/zhongjian_works/AI课题/施工进度估计/隧道施工项目/data_test/视频识别/渣土车识别/2025-11-28_110120_403.jpg"  # Update with your image path
    # file_path = "/Users/jinyfeng/个人文档/zhongjian_works/AI课题/施工进度估计/隧道施工项目/data_test/视频识别/渣土车识别/2025-11-28_110908_704.mp4"
    # Determine if the file is an image or video
    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
        # Process as video
        truck_number, ret_frame, timestamp = recognizer.recognize_trucknum_from_rtsp(file_path)
        # print(f"Results: {truck_number}")

    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        # Process as image
        image_path = file_path
    
        frame = cv2.imread(image_path)
        truck_number, ret_frame, timestamp = recognizer.detect_truck(frame)
        # print(f"Results: {results}")
        # cv2.imshow("Result", ret_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


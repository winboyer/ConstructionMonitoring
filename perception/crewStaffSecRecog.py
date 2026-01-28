import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional
import cv2
from PIL import Image, ImageDraw, ImageFont
import os, yaml, time
import onnxruntime
import sys
from ultralytics import YOLO
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perception.util import getDeteBBox_v2,getCropImg,getClsResult
from utils.image_process import (scale_person_bbox)
import base64
import requests
onnxruntime.set_default_logger_severity(3)
textFont = cv2.FONT_HERSHEY_SIMPLEX

class CrewStaffSecurityRecognizer:
    def __init__(self, person_model, safe_model):
        """
        Initialize the security client.
        """
        self.person_model = YOLO(person_model)
        self.safe_model = YOLO(safe_model)

    def person_detect(self, frame):
        results = self.person_model.predict(source=frame)

        cls = results[0].boxes.cls.cpu().numpy()
        cls_names = [self.person_model.names[int(c)] for c in cls]
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        ret_bboxes = []
        for i, name in enumerate(cls_names):
            if name == 'person':
                x1, y1, x2, y2 = map(int, bboxes[i])
                conf = confs[i]
                ret_bboxes.append((x1, y1, x2, y2, conf))
        return ret_bboxes

    def _detect_safety_gear(self, frame):
        """
        Detect safety gear in frame (placeholder for actual detection model).
        
        Returns:
            Tuple of (has_helmet, has_reflective_vest)
        """
        bboxes = self.person_detect(frame)
        print(f"detection bbox: {bboxes}")
        
        img_height, img_width = frame.shape[:2]
        scale = 1.2
        helmet_num, fgy_num = 0, 0
        for box in bboxes:
            x_min, y_min, x_max, y_max, box_conf = box
            x_min_new, y_min_new, x_max_new, y_max_new = scale_person_bbox(box, img_width, img_height, scale)

            person_crop = frame[int(y_min_new):int(y_max_new), int(x_min_new):int(x_max_new)]
            safe_results = self.safe_model.predict(source=person_crop)

            helmet_flag, fgy_flag = False, False
            for safe_result in safe_results:
                safe_cls = safe_result.boxes.cls.cpu().numpy()
                safe_conf = safe_result.boxes.conf.cpu().numpy()
                safe_boxes = safe_result.boxes.xyxy.cpu().numpy()
                safe_cls_names = [self.safe_model.names[int(c)] for c in cls]
                for idx, safe_name in enumerate(safe_cls_names):
                    if safe_name in ['aqm', 'fgmj']:
                        x_min_safe, y_min_safe, x_max_safe, y_max_safe = safe_boxes[idx]
                        conf = safe_boxes[idx]
                        print(f"Confidence: {conf}")
                        if conf < 0.5:
                            continue
                        x_min_orig = int(x_min_new + x_min_safe)
                        y_min_orig = int(y_min_new + y_min_safe)
                        x_max_orig = int(x_min_new + x_max_safe)
                        y_max_orig = int(y_min_new + y_max_safe)

                        if safe_name == 'aqm' and helmet_flag == False:
                            helmet_flag = True
                            helmet_num += 1
                            cv2.rectangle(frame, (x_min_orig, y_min_orig), (x_max_orig, y_max_orig), (0,0,255), 2)
                            # cv2.putText(frame, safe_name, (x_min_orig, y_min_orig-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,12,255), 2)
                        elif safe_name == 'fgmj' and fgy_flag == False:
                            fgy_flag = True
                            fgy_num += 1
                            cv2.rectangle(frame, (x_min_orig, y_min_orig), (x_max_orig, y_max_orig), (0,255,0), 2)
                            # cv2.putText(frame, safe_name, (x_min_orig, y_min_orig-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        no_helmet_num = len(bboxes) - helmet_num
        no_fgy_num = len(bboxes) - fgy_num
        
        return no_helmet_num, no_fgy_num, frame
    
    def recognize_security_from_rtsp(self, rtsp_url: str, interval=20) -> Tuple[bool, str]:
        """
        
        Args:
        
        Returns:
        """
        try:
            channel_name = rtsp_url.split('/')[-3]
            cap = None
            max_retries = 3
            for attempt in range(max_retries):
                cap = cv2.VideoCapture(rtsp_url)
                if cap.isOpened():
                    break
                print(f"Failed to open video source: {rtsp_url} (Attempt {attempt + 1}/{max_retries})")
                cap.release()
                time.sleep(2)  # Wait before retrying
            
            if cap is None or not cap.isOpened():
                print(f"Failed to open video source after {max_retries} attempts: {rtsp_url}")
                if cap is not None:
                    cap.release()
                return 
            
            image_save_folder = Path("detected_safety_violations")
            if not image_save_folder.exists():
                # image_save_folder.mkdir(parents=True, exist_ok=True)
                os.makedirs(image_save_folder, exist_ok=True)

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to read crewStaffSecRecog frame")
                    time.sleep(1)
                    continue
                
                timestamp = datetime.now().isoformat()
                dt = datetime.now().strftime('%Y%m%d_%H-%M-%S')
                t = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                no_helmet_num, no_fgy_num, ret_frame = self._detect_safety_gear(frame)
                if 'c13' in rtsp_url:
                    place = "工地大门口"
                    # time.sleep(1)
                elif 'c16' in rtsp_url:
                    place = "下井通道"
                    # time.sleep(2)

                filename = f"{channel_name}_frame_{timestamp}.jpg"
                temp_image_path = f"{image_save_folder}/{filename}"
                print(f"Temporary image path: {temp_image_path}")

                cv2.imwrite(temp_image_path, frame)
                if no_helmet_num > 0 or no_fgy_num > 0:
                    # cv2.imwrite(temp_image_path, frame)
                    # Convert frame to base64
                    # _, buffer = cv2.imencode('.jpg', ret_frame)
                    # frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    image_bytes = cv2.imencode('.jpg', ret_frame)[1].tobytes()
                    files={'imageFile':(filename, image_bytes, 'image/jpeg')}
                    url = "http://112.124.54.138:5001/api/safeInfo"
                    
                    response = requests.post(
                        url,
                        data={
                            "place": place,
                            "noClothesNum": no_fgy_num,
                            "noHelmetNum": no_helmet_num
                        },
                        files=files
                    )
                    print(f"Posted safety info to server: {response.status_code}, {response.text}")
                    time.sleep(interval)

                time.sleep(interval/10)
            
        except Exception as e:
            cap.release()
            print(f"Error during recognition: {str(e)}")
            return 

if __name__ == "__main__":
    model_path = "/Users/jinyfeng/tools/helmet"  # Update with your model directory
    recognizer = CrewStaffSecurityRecognizer(model_path)
    
    # image_path = "/Users/jinyfeng/个人文档/zhongjian_works/AI课题/施工进度估计/隧道施工项目/2.jpg"
    image_path = "/Users/jinyfeng/个人文档/zhongjian_works/AI课题/施工进度估计/隧道施工项目/20251226093810_522_154.png"  # Update with your image path

    image_folder = "/Users/jinyfeng/projects/ConstructionMonitoring/safe_images/dixia/"

    model_path = "/Users/jinyfeng/tools/object-det/yolo11s.pt"  # Update with your model directory
    det_model = YOLO(model_path)

    image_lists = os.listdir(image_folder)
    for image_name in image_lists:
        print(f"Processing image: {image_name}")
        image_path = os.path.join(image_folder, image_name)

        image = cv2.imread(image_path)
        results = det_model.predict(source=image)

        xyxy = results[0].boxes.xyxy.cpu().numpy()
        xywh = results[0].boxes.xywh.cpu().numpy()
        conf = results[0].boxes.conf.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy()

        class_names = det_model.names

        print("Detected objects:")
        scale = 1.5  # 放大比例
        for i, c in enumerate(cls):
            print(f"Class: {class_names[int(c)]}")
            if class_names[int(c)] == 'person':
                
                # print("Found a person, enlarging bbox...")
                # print(f"Original bbox: {xyxy[i]}")
                x1, y1, x2, y2 = xyxy[i]
                # print("Original xywh:", xywh[i])
                # 计算中心点
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 计算原始宽度和高度
                width = x2 - x1
                height = y2 - y1
                
                # 计算新的宽度和高度
                new_width = width * scale * 2
                new_height = height * scale
                
                # 计算新的边界框坐标
                new_x1 = center_x - new_width / 2
                new_y1 = center_y - new_height / 2
                new_x2 = center_x + new_width / 2
                new_y2 = center_y + new_height / 2

                # frame = image[int(xyxy[i][1]):int(xyxy[i][3]), int(xyxy[i][0]):int(xyxy[i][2])] # Crop original bbox with y1, y2, x1, x2
                frame = image[int(new_y1):int(new_y2), int(new_x1):int(new_x2)] # Crop original bbox with y1, y2, x1, x2
                
                cv2.imshow("frame", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                no_helmet_num, no_fgy_num, ret_frame = recognizer._detect_safety_gear(frame)
                print(f"No helmet detected: {no_helmet_num}, No reflective vest detected: {no_fgy_num}")


        
        # if frame is None:
        #     print(f"Failed to load image: {image_path}")
        # else:
        #     no_helmet_num, no_fgy_num, ret_frame = recognizer._detect_safety_gear(frame)
        #     print(f"No helmet detected: {no_helmet_num}, No reflective vest detected: {no_fgy_num}")

        #     cv2.imshow("Result", ret_frame)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()


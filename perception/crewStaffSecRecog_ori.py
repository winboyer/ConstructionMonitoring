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
import base64
import requests
onnxruntime.set_default_logger_severity(3)
textFont = cv2.FONT_HERSHEY_SIMPLEX

class CrewStaffSecurityRecognizer:
    def __init__(self, model_path):
        """
        Initialize the security client.
        """
        dete_onnx_path = os.path.join(model_path, 'onnx/ppyoloe_plus_sod_0823.onnx')
        cls_onnx_path = os.path.join(model_path, 'onnx/cls-4.onnx')
        dete_cfg_path = os.path.join(model_path, 'onnx/infer_cfg.yml')
        with open(dete_cfg_path) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.draw_threshold = yml_conf.get("draw_threshold", 0.4)

        self.deteModel = onnxruntime.InferenceSession(dete_onnx_path, providers=["CPUExecutionProvider"])
        self.clsModel = onnxruntime.InferenceSession(cls_onnx_path, providers=["CPUExecutionProvider"])

    def _detect_safety_gear(self, frame):
        """
        Detect safety gear in frame (placeholder for actual detection model).
        
        Returns:
            Tuple of (has_helmet, has_reflective_vest)
        """
        bbox = getDeteBBox_v2(self.preprocess_infos, self.draw_threshold, self.deteModel, frame)
        print(f"detection bbox: {bbox}")
        
        out_helmet, out_fgy = True, True          # Default to True
        no_helmet_num, no_fgy_num = 0, 0
        if len(bbox) != 0:
            batch_crop_img = getCropImg(frame, bbox)
            result_list = getClsResult(batch_crop_img, self.clsModel)
            
            # print(result_list, bbox) # 反光衣【0,1】 安全帽【0,1,2看不到】 长短袖【0长 1短 2未知】 人 【0,1】
            # print(len(bbox), len(result_list))
            for bb, r in zip(bbox, result_list):
                cls, conf, x1, y1, x2, y2 = bb
                # print(x1, y1, x2, y2)
                cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 2)
                pred_fgy = r[0]
                pred_helmet = r[1]
                pred_person = r[3]
                # print(pred_fgy,pred_helmet,pred_person)
                out_fgy = False if pred_fgy[0] >= 0.8 else True
                out_helmet = False if pred_helmet[0] >= 0.8 else True
                out_person = True if pred_person[1] >= 0.8 else False
                # print(f"out_helmet: {out_helmet}, out_fgy: {out_fgy}, out_person: {out_person}")
                if out_helmet == False and out_person == True:
                    # frame = cv2.putText(frame, 'no helmet', (x1, y1-10), textFont, 1, (0,0,255), 1)
                    # cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 5)
                    print('未戴安全帽')
                    no_helmet_num += 1
                if out_fgy == False and out_person == True:
                    # cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 5)
                    # frame = cv2.putText(frame, 'no reflect_vest', (x1, (y1+y2)//2), textFont, 1, (0,255,0), 1)
                    print('未穿反光衣')
                    no_fgy_num += 1
        
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
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to read crewStaffSecRecog frame")
                    time.sleep(1)
                    continue

                image_save_folder = Path("detected_safety_violations")
                if not image_save_folder.exists():
                    # image_save_folder.mkdir(parents=True, exist_ok=True)
                    os.makedirs(image_save_folder, exist_ok=True)
                
                if 'c13' in rtsp_url:
                    place = "工地大门口"
                    time.sleep(1)
                else:
                    place = "下井通道"
                    time.sleep(2)

                no_helmet_num, no_fgy_num, ret_frame = self._detect_safety_gear(frame)
                
                timestamp = datetime.now().isoformat()
                dt = datetime.now().strftime('%Y%m%d_%H-%M-%S')
                t = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

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


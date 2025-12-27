import requests
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional
import cv2
from PIL import Image, ImageDraw, ImageFont
import os, yaml, time
import onnxruntime
from util import getDeteBBox_v2,getCropImg,getClsResult
import base64
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
        out_helmet, out_fgy = False, False  # Default to True
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
                    frame = cv2.putText(frame, 'no helmet', (x1, y1-10), textFont, 1, (0,0,255), 1)
                    # cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 5)
                    print('未戴安全帽')
                if out_fgy == False and out_person == True:
                    # cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 5)
                    frame = cv2.putText(frame, 'no reflect_vest', (x1, (y1+y2)//2), textFont, 1, (0,255,0), 1)
                    print('未穿反光衣')
        
        return out_helmet, out_fgy, frame
    
    def recognize_security_from_rtsp(self, rtsp_url: str, interval=20) -> Tuple[bool, str]:
        """
        
        Args:
        
        Returns:
        """
        try:
            
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print(f"Failed to open video source: {rtsp_url}")
                cap.release()
                return False, "Failed to open video source"
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to read frame")
                    continue

                helmet_flag, reflectvest_flag, ret_frame = self._detect_safety_gear(frame)
                if helmet_flag is False or reflectvest_flag is False:
                    timestamp = datetime.now().isoformat()
                    temp_image_path = f"violation_frame_{timestamp}.jpg"
                    cv2.imwrite(temp_image_path, frame)
                    # Convert frame to base64
                    _, buffer = cv2.imencode('.jpg', ret_frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    cap.release()
                    return helmet_flag, reflectvest_flag, frame_base64, timestamp

                time.sleep(interval)                
            
        except Exception as e:
            cap.release()
            return False, f"Error: {str(e)}"



if __name__ == "__main__":
    model_path = "/Users/jinyfeng/tools/helmet"  # Update with your model directory
    recognizer = CrewStaffSecurityRecognizer(model_path)
    
    # image_path = "/Users/jinyfeng/个人文档/zhongjian_works/AI课题/施工进度估计/隧道施工项目/2.jpg"
    image_path = "/Users/jinyfeng/个人文档/zhongjian_works/AI课题/施工进度估计/隧道施工项目/20251226093810_522_154.png"  # Update with your image path

    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Failed to load image: {image_path}")
    else:
        helmet_flag, reflectvest_flag, ret_frame = recognizer._detect_safety_gear(frame)
        print(f"Helmet detected: {helmet_flag}, Reflective vest detected: {reflectvest_flag}")
        cv2.imshow("Result", ret_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


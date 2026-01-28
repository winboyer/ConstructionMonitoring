import re
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional
import cv2
import sys
import os
import time
from ultralytics import YOLO


def object_detect(frame, model):
    
    results = model.predict(source=frame)

    xyxy = results[0].boxes.xyxy.cpu().numpy()
    xywh = results[0].boxes.xywh.cpu().numpy()
    conf = results[0].boxes.conf.cpu().numpy()
    cls = results[0].boxes.cls.cpu().numpy()                

    class_names = model.names
    # print('class_names:', class_names)
    detected_objects = []

    # frame_copy = frame.copy()
    truck_number = None
    timestamp = None

    print("Detected objects:")
    for i, c in enumerate(cls):
        print(f"Class: {class_names[int(c)]}")
        
        if class_names[int(c)] == 'person':
            print("Detected a person!")
            # cv2.rectangle(frame, (int(xyxy[i][0]), int(xyxy[i][1])), (int(xyxy[i][2]), int(xyxy[i][3])), (0, 255, 0), 2)
            
            # cv2.imshow("Frame", frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return frame

    return None



if __name__ == "__main__":
    
    # folder = "/Users/jinyfeng/Downloads/"
    folder = "/home/jinyfeng/datas/suidao/"
    folder = "/home/jinyfeng/datas/suidao/safe_det/xiajingkou16_20260120/"
    folder = "/home/jinyfeng/datas/suidao/safe_det/xiajingkou16_20260121/"

    # folder = "/home/jinyfeng/datas/suidao/guanpian_det/20260126/"

    # filename = "2897f7c79704abf07bf74d05ed0e585a.mp4"
    # filename = "2897f7c79704abf07bf74d05ed0e585a_raw.mp4"
    filename = "16f3d2dbf72b7506c8252dcf147f6758.mp4"
    # filename = "16f3d2dbf72b7506c8252dcf147f6758_raw.mp4" 

    # filename = 'S20260126170251_E20260126170519.mp4'
    # filename = 'S20260126171822_E20260126172102.mp4'
    # filename = 'S20260126172820_E20260126173118.mp4'
    filename = 'S20260126174446_E20260126174750.mp4'

    filename = "S20260120162627_E20260120163550.mp4" # xiajingkou16_20260120

    filename = "S20260121083137_E20260121083223.mp4" # xiajingkou16_20260121
    filename = "S20260121083224_E20260121083444.mp4" # xiajingkou16_20260121
    filename = "S20260121142254_E20260121143041.mp4" # xiajingkou16_20260121
    filename = "S20260121145338_E20260121145633.mp4" # xiajingkou16_20260121
    filename = "S20260121145917_E20260121150608.mp4" # xiajingkou16_20260121

    file_path = os.path.join(folder, filename)
    save_folder = os.path.join(folder, filename.split('.')[0])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # model_path = "/Users/jinyfeng/tools/object-det/yolo11n.pt"  # Update with your model directory
    model_path = "/home/jinyfeng/models/yolo/yolo-v11/yolo11n.pt"  # Update with your model directory
    det_model = YOLO(model_path)

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    cap = cv2.VideoCapture(file_path)
    image_counter = 0
    frame_cnt = 0
    if cap is None or not cap.isOpened():
        print(f"Failed to open video source: {file_path}")
        cap.release()
        sys.exit(1)
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print(f"Failed to read frame")
            time.sleep(1)
            break
        # print(frame.shape)
        # 间隔保存每第三帧
        frame_cnt += 1
        if frame_cnt % 9 != 0:
            continue

        # image_counter += 1
        # print(f"Saving image {image_counter}")
        # save_name = os.path.join(save_folder, f"{filename.split('.')[0]}_cnt_{image_counter}.jpg")
        # cv2.imwrite(save_name, frame)
        # time.sleep(0.01)
        
        result = object_detect(frame, det_model)
        if result is not None:
            print("Person detected and image saved.")
            # Optionally break the loop if you only want the first detection
            image_counter += 1
            print(f"Saving image {image_counter}")
            save_name = os.path.join(save_folder, f"{filename.split('.')[0]}_cnt_{image_counter}.jpg")
            cv2.imwrite(save_name, frame)
            time.sleep(0.01)
            

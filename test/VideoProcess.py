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
            cv2.rectangle(frame, (int(xyxy[i][0]), int(xyxy[i][1])), (int(xyxy[i][2]), int(xyxy[i][3])), (0, 255, 0), 2)

            cv2.imshow("Frame", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return cls, frame



if __name__ == "__main__":
    
    file_path = "/Users/jinyfeng/Downloads/2897f7c79704abf07bf74d05ed0e585a_raw.mp4"  # Update with your video file path
    # file_path = "/Users/jinyfeng/Downloads/16f3d2dbf72b7506c8252dcf147f6758_raw.mp4"  # Update with your video file path
    model_path = "/Users/jinyfeng/tools/object-det/yolo11n.pt"  # Update with your model directory
    det_model = YOLO(model_path)

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    cap = cv2.VideoCapture(file_path)
    if cap is None or not cap.isOpened():
        print(f"Failed to open video source: {file_path}")
        cap.release()
        sys.exit(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame")
            time.sleep(1)
            continue
        
        results, frame = object_detect(frame, det_model)

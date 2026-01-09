import cv2
import threading
import time
import requests
import json
from datetime import datetime
import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_process import bytes_to_base64, base64_to_bytes
from perception.crewStaffSecRecog import CrewStaffSecurityRecognizer
from perception.truckRecog import TruckRecognizer

class RTSPStreamRecognizer:
    def __init__(self, rtsp_urls, truck_model, safe_model, interval=20):
        """
        Args:
            rtsp_urls: List of 16 RTSP stream URLs
            truck_model: Path to the truck detection model
            interval: Capture frame interval in seconds (default: 20s)
        """
        self.rtsp_urls = rtsp_urls
        self.interval = interval
        self.threads = []
        self.running = True
        self.truck_recognizer = TruckRecognizer(truck_model)
        self.safe_recognizer = CrewStaffSecurityRecognizer(safe_model)
    
    def process_stream(self, rtsp_url):
        """Process single RTSP stream"""
        
        # Determine interval based on stream ID
        # 安全帽和反光衣识别
        if 'c16' in rtsp_url or 'c13' in rtsp_url:
            interval = 3
            noHelmetNum, noClothesNum, filename, ret_frame = self.safe_recognizer.recognize_security_from_rtsp(rtsp_url)
            image_bytes = cv2.imencode('.jpg', ret_frame)[1].tobytes()
            files={'imageFile':(filename, image_bytes, 'image/jpeg')}
            url = "http://112.124.54.138:5001/api/safeInfo"
            if 'c13' in rtsp_url:
                place = "工地大门口"
                time.sleep(1)
            else:
                place = "下井通道"
                time.sleep(2)
            response = requests.post(
                url,
                data={
                    "place": place,
                    "noClothesNum": noClothesNum,
                    "noHelmetNum": noHelmetNum
                },
                files=files
            )
            print(f"Posted safety info to server: {response.status_code}, {response.text}")

        # elif any(cam in rtsp_url for cam in ['c11', 'c13', 'c23', 'c25', 'c26']):
        #     interval = self.interval
        # c4 渣土车车牌号识别
        elif 'c4' in rtsp_url:
            interval = 300
            truck_number, ret_frame, timestamp = self.truck_recognizer.recognize_trucknum_from_rtsp(rtsp_url)
            # ret_frame_b64 = bytes_to_base64(image_bytes)
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
            else:
                time.sleep(interval//2)
        else:
            interval = self.interval
                
    def start(self):
        """Start processing all streams"""
        for rtsp_url in self.rtsp_urls:
            thread = threading.Thread(
                target=self.process_stream,
                args=(rtsp_url,),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)        
        print(f"Started processing {len(self.rtsp_urls)} streams")
        # Keep main thread alive
        try:
            while True:
                time.sleep(self.interval)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop all streams"""
        self.running = False
        for thread in self.threads:
            thread.join()
        print("All streams stopped")


if __name__ == "__main__":
    # Configure your 16 RTSP URLs here
    rtsp_urls = [
        # c11/c13/c16/c23/c25/c26进行安全帽和反光衣识别，先处理C13/C16
        # c4 渣土车车牌号识别, (c13)也可，但角度不固定
        # "rtsp://admin:123456@192.168.110.118:554/unicast/c2/s0/live",	        #盾构机枪机	台车尾部
        # "rtsp://admin:123456@192.168.110.118:554/unicast/c3/s0/live"	        #盾构机枪机	皮带出渣口
	    # "rtsp://admin:123456@192.168.110.118:554/unicast/c5/s0/live"	        #盾构机枪机	砂浆罐
	    # "rtsp://admin:123456@192.168.110.118:554/unicast/c6/s0/live"	        #盾构机枪机	拼装区
	    # "rtsp://admin:123456@192.168.110.118:554/unicast/c7/s0/live"	        #盾构机枪机	人仓主仓
	    # "rtsp://admin:123456@192.168.110.118:554/unicast/c8/s0/live"	        #盾构机枪机	1号仓副仓
	    # "rtsp://admin:123456@192.168.110.118:554/unicast/c9/s0/live"	        #盾构机枪机	出料口
	    # "rtsp://admin:zgjz@office@192.168.110.2:554/h264/ch1/main/av_stream"	#盾构机枪机	盾构司机室
	    
        # "rtsp://admin:123456@192.168.110.118:554/unicast/c1/s0/live"	        #外部枪机	监控室
	    "rtsp://admin:123456@192.168.110.127:554/unicast/c4/s0/live"	        #外部枪机	工地大门口1
	    # "rtsp://admin:123456@192.168.110.118:554/unicast/c10/s0/live"	        #外部球机	砂浆站
	    # "rtsp://admin:123456@192.168.110.118:554/unicast/c11/s0/live"	        #外部球机	右线概况
	    # "rtsp://admin:123456@192.168.110.118:554/unicast/c12/s0/live"	        #外部球机	路口围挡
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c13/s0/live"	        #外部球机	工地大门口2
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c16/s0/live"	        #外部枪机	下井通道
	    # "rtsp://admin:123456@192.168.110.118:554/unicast/c22/s0/live"	        #外部球机	右线井口
	    # "rtsp://admin:123456@192.168.110.118:554/unicast/c23/s0/live"	        #外部球机	左线洞口
	    # "rtsp://admin:123456@192.168.110.118:554/unicast/c24/s0/live"	        #外部球机	左线井口
	    # "rtsp://admin:123456@192.168.110.118:554/unicast/c25/s0/live"	        #外部球机	左线概况
	    # "rtsp://admin:123456@192.168.110.118:554/unicast/c26/s0/live"	        #外部球机	右线洞口
    ]
    truck_model = "/Users/jinyfeng/tools/object-det/yolo11n.pt"
    recognizer = RTSPStreamRecognizer(rtsp_urls, truck_model, interval=20)
    recognizer.start()
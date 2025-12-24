import cv2
import threading
import time
from paddleocr import PaddleOCR
from datetime import datetime

class RTSPStreamRecognizer:
    def __init__(self, rtsp_urls, interval=20):
        """
        Args:
            rtsp_urls: List of 16 RTSP stream URLs
            interval: Capture frame interval in seconds (default: 20s)
        """
        self.rtsp_urls = rtsp_urls
        self.interval = interval
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        self.threads = []
        self.running = True
    
    def process_stream(self, stream_id, rtsp_url):
        """Process single RTSP stream"""
        cap = cv2.VideoCapture(rtsp_url)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"Stream {stream_id}: Failed to read frame")
                time.sleep(5)
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)
                continue
            
            # Perform OCR
            results = self.ocr.ocr(frame, cls=True)
            
            # Process results
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[Stream {stream_id}] {timestamp}")
            for line in results:
                for word_info in line:
                    text = word_info[1][0]
                    confidence = word_info[1][1]
                    print(f"  Text: {text} | Confidence: {confidence:.2f}")
            
            # Wait for next capture
            time.sleep(self.interval)
        
        cap.release()
    
    def start(self):
        """Start processing all streams"""
        for stream_id, rtsp_url in enumerate(self.rtsp_urls, 1):
            thread = threading.Thread(
                target=self.process_stream,
                args=(stream_id, rtsp_url),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        print(f"Started processing {len(self.rtsp_urls)} streams")
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
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
        "rtsp://admin:123456@192.168.110.118:554/unicast/c2/s0/live",	        #盾构机枪机	台车尾部
        "rtsp://admin:123456@192.168.110.118:554/unicast/c3/s0/live"	        #盾构机枪机	皮带出渣口
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c5/s0/live"	        #盾构机枪机	砂浆罐
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c6/s0/live"	        #盾构机枪机	拼装区
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c7/s0/live"	        #盾构机枪机	人仓主仓
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c8/s0/live"	        #盾构机枪机	1号仓副仓
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c9/s0/live"	        #盾构机枪机	出料口
	    "rtsp://admin:zgjz@office@192.168.110.2:554/h264/ch1/main/av_stream"	#盾构机枪机	盾构司机室
	    
        "rtsp://admin:123456@192.168.110.118:554/unicast/c1/s0/live"	        #外部枪机	监控室
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c4/s0/live"	        #外部枪机	工地大门口1

	    "rtsp://admin:123456@192.168.110.118:554/unicast/c10/s0/live"	        #外部球机	砂浆站
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c11/s0/live"	        #外部球机	右线概况
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c12/s0/live"	        #外部球机	路口围挡
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c13/s0/live"	        #外部球机	工地大门口2
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c16/s0/live"	        #外部枪机	下井通道
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c22/s0/live"	        #外部球机	右线井口
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c23/s0/live"	        #外部球机	左线洞口
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c24/s0/live"	        #外部球机	左线井口
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c25/s0/live"	        #外部球机	左线概况
	    "rtsp://admin:123456@192.168.110.118:554/unicast/c26/s0/live"	        #外部球机	右线洞口
    ]
    
    recognizer = RTSPStreamRecognizer(rtsp_urls, interval=20)
    recognizer.start()
import json
import requests
import time
import base64
from datetime import datetime
import cv2


def test_captcha_service():
    url = "http://172.30.4.220:2512/captcha/recognize"
    # url = "http://172.20.10.5:2512/captcha/recognize"
    # url = "http://172.30.65.194:2512/captcha/recognize"
    # url = "http://112.124.54.138:2512/captcha/recognize"
    # url = "http://192.168.7.157:2512/captcha/recognize"
    
    response = requests.get(url, timeout=5)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")


def test_deliver_doc_service():
    url = "http://172.30.4.220:5000/recognize"
    # image_path = "output.png"
    # image_path = "/Users/jinyfeng/Downloads/微信图片_20251106160955_276_154.jpg"
    image_path = '/Users/jinyfeng/Downloads/data_test/1111111_bce4f761-29e9-43ff-8559-1eb11231a5e4.png'
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    response = requests.post(url, json={'image': image_base64})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")


def test_truckPlate_service():
    url = "http://112.124.54.138:5001/api/loadcars/image"
    # image_path = "output.png"
    image_path = './detected_trucks/image_20251229_120459.jpg'
    # with open(image_path, "rb") as image_file:
    #     image_bytes = image_file.read()
    img_name = image_path.split('/')[-1]
    
    image = cv2.imread(image_path)
    _, img_encoded = cv2.imencode('.jpg', image)
    image_bytes = img_encoded.tobytes()
    
    # files={'imageFile':(img_name, open(image_path,'rb'),'image/jpeg')}
    files={'imageFile':(img_name, image_bytes, 'image/jpeg')}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    response = requests.post(
        url,
        data={
            "licenSeplate": "辽CC3171",
            "timestamp": timestamp
        },
        files=files
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def test_safeInfo_service():
    url = "http://112.124.54.138:5001/api/safeInfo"
    # image_path = "output.png"
    # image_path = "/Users/jinyfeng/Downloads/微信图片_20251106160955_276_154.jpg"
    # image_path = '/Users/jinyfeng/Downloads/data_test/1111111_bce4f761-29e9-43ff-8559-1eb11231a5e4.png'
    image_path = '/Users/jinyfeng/个人文档/zhongjian_works/AI课题/施工进度估计/隧道施工项目/data_test/1.jpg'
    img_name = image_path.split('/')[-1]

    image = cv2.imread(image_path)
    image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
    files={'imageFile':(img_name, image_bytes, 'image/jpeg')}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    noClothesNum = 1
    noHelmetNum = 1
    response = requests.post(
        url,
        data={
            "place": "test",
            "noClothesNum": noClothesNum,
            "noHelmetNum": noHelmetNum
        },
        files=files
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")


if __name__ == "__main__":
    # test_captcha_service()
    # test_deliver_doc_service()
    # test_truckPlate_service()
    test_safeInfo_service()


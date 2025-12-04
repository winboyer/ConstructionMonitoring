import cv2
from paddlex import create_model
from pyzbar import pyzbar

def detect_and_decode_qrcodes(image_path):
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图片:", image_path)
        return

    model = create_model(model_name = 'PP-LCNet_x1_0')
    output = model.predict(image_path, batch_size=1)
    print("模型预测结果:", output)



    # 检测并解码二维码
    qrcodes = pyzbar.decode(image)
    for qr in qrcodes:
        # 获取二维码位置
        (x, y, w, h) = qr.rect
        # 绘制矩形框
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 输出二维码内容
        qr_data = qr.data.decode('utf-8')
        qr_type = qr.type
        print(f"检测到二维码: 类型={qr_type}, 内容={qr_data}")
        # 在图片上显示内容
        cv2.putText(image, qr_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示结果图片
    cv2.imshow("QR Code Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 替换为你的图片路径
    image_path = "/home/common_datas/jinyfeng/datas/data_test/视频识别/管片识别/MVIMG_20251127_092850.jpg"
    detect_and_decode_qrcodes(image_path)
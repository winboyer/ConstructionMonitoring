import argparse
import requests
import cv2
import logging
import time
import onnxruntime
import os, yaml
from concurrent.futures import ThreadPoolExecutor
from util import getDeteBBox, getCropImg, getClsResult
from datetime import datetime, timedelta

onnxruntime.set_default_logger_severity(3)


def init_logger():
    LOG_FORMAT = '[AiServer][%(asctime)s %(levelname)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, filename=r'helmet.log', filemode='w+')


class DeteConfig(object):
    def __init__(self, infer_config):
        # parsing Yaml config for Preprocess
        with open(infer_config) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.draw_threshold = yml_conf.get("draw_threshold", 0.5)


def model_load(args):
    logging.info("Start Loading model with gpu mode...")
    dete_onnx_path = os.path.join(args.model_path, 'onnx/ppyoloe_plus_sod_0823.onnx')
    cls_onnx_path = os.path.join(args.model_path, 'onnx/cls-4.onnx')
    dete_cfg_path = os.path.join(args.model_path, 'onnx/infer_cfg.yml')
    dete_config = DeteConfig(dete_cfg_path)
    deteModel = onnxruntime.InferenceSession(dete_onnx_path, providers=["CPUExecutionProvider"])
    clsModel = onnxruntime.InferenceSession(cls_onnx_path, providers=["CPUExecutionProvider"])
    return deteModel, dete_config, clsModel


def video_frame_reco(deteModel, dete_config, clsModel, rtsp, model_path):
    last_count_time = datetime(2024, 5, 24, 10, 0, 0)
    while True:
        if datetime.now() - last_count_time >= timedelta(minutes=2):
            logging.info(f"Start dealing rtsp video streaming,{rtsp}")
            try:
                cap = cv2.VideoCapture(rtsp)
                _, frame = cap.read()
                bbox = getDeteBBox(dete_config, deteModel, frame)
                if len(bbox) != 0:
                    batch_crop_img = getCropImg(frame, bbox)
                    result_list = getClsResult(batch_crop_img, clsModel)
                    
                    for bb, r in zip(bbox, result_list):
                        cls, conf, x1, y1, x2, y2 = bb
                        # print(x2-x1,y2-y1)

                        pred_fgy = r[0]
                        pred_helmet = r[1]
                        pred_person = r[3]

                        out_fgy = 0 if pred_fgy[0] >= 0.8 else 1
                        out_helmet = 0 if pred_helmet[0] >= 0.8 else 1
                        out_person = 1 if pred_person[1] >= 0.8 else 0
                        if out_helmet == 0 and out_person == 1:
                            logging.info('未戴安全帽')
                            current_time = datetime.now().strftime("%Y%m%d%H%M")
                            img_name_ori = os.path.join(model_path,
                                                        f'save/{current_time}_ori.jpg')
                            cv2.imwrite(img_name_ori, frame)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
                            img_name = os.path.join(model_path,
                                                    f'save/{current_time}.jpg')
                            cv2.imwrite(img_name, frame)
                            # 将图片及告警上传到云服务
                            upload_alarm_server('未戴安全帽', img_name,rtsp,4)
                            logging.info(f"The image {img_name} upload")
                        if out_fgy == 0 and out_person == 1:
                            logging.info('未穿反光衣')
                            current_time = datetime.now().strftime("%Y%m%d%H%M")
                            img_name_ori = os.path.join(model_path,
                                                        f'save/{current_time}_ori.jpg')
                            cv2.imwrite(img_name_ori, frame)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
                            img_name = os.path.join(model_path,
                                                    f'save/{current_time}.jpg')
                            cv2.imwrite(img_name, frame)
                            # 将图片及告警上传到云服务
                            upload_alarm_server('未穿反光衣', img_name, rtsp,7)
                            logging.info(f"The image {img_name} upload")

                cap.release()
                logging.info(f"END dealing rtsp video streaming,{rtsp}")
                last_count_time = datetime.now()
            except Exception as e:
                logging.error(f"{rtsp} streaming read error,{e}")
                last_count_time = datetime.now()


def upload_alarm_server(alarm_content, input_img_path, rtsp, type_index):
    try:
        logging.info("Start uploading")
        file = open(input_img_path, "rb")
        files = {"attachment": file}
        time_now = time.strftime("%Y-%m-%d %H:%M:%S")
        # number为摄像头ip
        params = {"area": "1970314898858020865", "content": alarm_content, "number": rtsp, "time": time_now,
                  "type": type_index, 'business': 1}
        response = requests.post(url='https://szdtmap.cn:10000/api/skip/earlyWarning/save', files=files, params=params)
        if response.status_code == 200:
            logging.info("upload  success")
        file.close()
    except Exception as e:
        logging.error(f"upload failed:{e}")



def main(args):
    init_logger()
    deteModel, dete_config, clsModel = model_load(args)
    with ThreadPoolExecutor(max_workers=len(args.rtsps)) as executor:
        for rtsp in args.rtsps:
            executor.submit(video_frame_reco, deteModel, dete_config, clsModel, rtsp, args.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=False, default=r'/home/tl/aiserver/helmet')
    parser.add_argument("--rtsps", type=str, required=False,
                        default=['rtsp://111.172.230.138:38799/rtp/34020000001320000001_34020000001320000037',
                                 'rtsp://111.172.230.138:38799/rtp/34020000001320000002_34020000001320000035'])
    args = parser.parse_args()
    main(args)

import argparse
import requests
import cv2
import logging
import time
import onnxruntime
import os,yaml
from util import getDeteBBox,getCropImg,getClsResult
import numpy as np
onnxruntime.set_default_logger_severity(3)
font = cv2.FONT_HERSHEY_SIMPLEX

class DeteConfig(object):
    """set config of dete preprocess, postprocess and visualize
    Args:
        infer_config (str): path of infer_cfg.yml
    """

    def __init__(self, infer_config):
        # parsing Yaml config for Preprocess
        with open(infer_config) as f:
            yml_conf = yaml.safe_load(f)
        self.preprocess_infos = yml_conf['Preprocess']
        self.draw_threshold = yml_conf.get("draw_threshold", 0.3)


def model_load(argv):
    dete_onnx_path = os.path.join(argv.model_path,'onnx/ppyoloe_plus_sod_0823.onnx')
    cls_onnx_path = os.path.join(argv.model_path, 'onnx/cls-4.onnx')
    dete_cfg_path = os.path.join(argv.model_path,'onnx/infer_cfg.yml')
    dete_config = DeteConfig(dete_cfg_path)
    deteModel = onnxruntime.InferenceSession(dete_onnx_path,providers=["CPUExecutionProvider"])
    clsModel = onnxruntime.InferenceSession(cls_onnx_path,providers=["CPUExecutionProvider"])
    return deteModel,dete_config,clsModel

def video_frame_reco(argv):
    deteModel, dete_config, clsModel = model_load(argv)
    for name in os.listdir(argv.dir):
        #print(name)
        # name ='202510220303_ori.jpg'
        name = '/Users/jinyfeng/个人文档/zhongjian_works/AI课题/施工进度估计/隧道施工项目/2.jpg'
        try:
            filename = os.path.join(argv.dir,name)
            frame = cv2.imread(filename)
            #print(frame.shape)
            bboxes = getDeteBBox(dete_config, deteModel, frame)
            if len(bboxes)!=0:
                batch_crop_img = getCropImg(frame, bboxes)
                result_list = getClsResult(batch_crop_img, clsModel)
                #print(result_list,bbox) # 反光衣【0,1】 安全帽【0,1,2看不到】 长短袖【0长 1短 2未知】 人 【0,1】
                for bb, r in zip(bboxes, result_list):
                    cls, conf, x1, y1, x2, y2 = bb
                    h, w = y2 - y1, x2 - x1
                    scale_h, scale_w = int(h / 10), int(w / 10)
                    #print(x2-x1,y2-y1)
                    cv2.rectangle(frame, (x1-scale_w, y1-scale_h), (x2+scale_w, y2+scale_h), (255, 0, 0), 5)
                    pred_fgy = r[0]
                    pred_helmet = r[1]
                    pred_person = r[3]
                    print(pred_fgy,pred_helmet,pred_person)
                    out_fgy = 0 if pred_fgy[0] >= 0.8 else 1
                    out_helmet = 0 if pred_helmet[0] >= 0.8 else 1
                    out_person = 1 if pred_person[1] >= 0.8 else 0
                    # if out_helmet == 0 and out_person == 1:
                    #     #frame = cv2.putText(frame, 'no helmet:{:.2f}'.format(pred1[0]), (x1-10, y1-10), font, 1, (60,255,255), 2)
                    #     cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 5)
                    #     print('未戴安全帽')
                    if out_fgy == 0 and out_person == 1:
                        #frame = cv2.putText(frame, 'no helmet:{:.2f}'.format(pred1[0]), (x1-10, y1-10), font, 1, (60,255,255), 2)
                        cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 5)
                        print('未穿反光衣')
                    if out_helmet == 1 and out_person == 1:
                        #cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 255, 0), 5)
                        print('normal')

                cv2.imshow('1',frame)
                cv2.waitKey(0)
            else:
                print('no person')
        except:
            print(name)




def main(argv):
    video_frame_reco(argv)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=False, default='/Users/jinyfeng/tools/helmet/')
    parser.add_argument("--dir", type=str,default='/Users/jinyfeng/tools/')
    args = parser.parse_args()
    main(args)



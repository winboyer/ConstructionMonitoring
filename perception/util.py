import cv2
from perception.preprocess import Compose
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
input_size = (224, 224)

valid_trans = transforms.Compose(
	[transforms.Resize(input_size), transforms.ToTensor()])

def softmax(x):
	# 计算每个元素的指数
	exps = np.exp(x - np.max(x))  # 减去最大值以提高数值稳定性
	# 计算所有指数的和
	sum_exps = np.sum(exps)
	# 计算softmax值
	return exps / sum_exps
	
def getDeteBBox(infer_config, predictor,img):
	# 运行检测模型进行推理
	h,w,_ = img.shape
	transforms = Compose(infer_config.preprocess_infos)
	result = []
	inputs = transforms(img)
	inputs_name = [var.name for var in predictor.get_inputs()]
	inputs = {k: inputs[k][None, ] for k in inputs_name}
	outputs = predictor.run(output_names=None, input_feed=inputs)
	bboxes = np.array(outputs[0])
	conf_bboxes = bboxes[:, 1]
	#conf filter
	# 创建布尔索引
	filter_index = conf_bboxes > infer_config.draw_threshold
	# 使用布尔索引来选择数组中对应的行
	bboxes = bboxes[filter_index]
	for bbox in bboxes:
		if bbox[2] >= 5 and bbox[3] >= 5 and bbox[4] <= w-5 and bbox[5] <= h-5:
			result.append([int(bbox[0]),bbox[1],int(bbox[2]),int(bbox[3]),int(bbox[4]),int(bbox[5])])
	
	return result

def getDeteBBox_v2(preprocess_infos, draw_threshold, predictor,img):
	# 运行检测模型进行推理
	h,w,_ = img.shape
	transforms = Compose(preprocess_infos)
	result = []
	inputs = transforms(img)
	inputs_name = [var.name for var in predictor.get_inputs()]
	inputs = {k: inputs[k][None, ] for k in inputs_name}
	outputs = predictor.run(output_names=None, input_feed=inputs)
	bboxes = np.array(outputs[0])
	conf_bboxes = bboxes[:, 1]
	#conf filter
	# 创建布尔索引
	filter_index = conf_bboxes > draw_threshold
	# 使用布尔索引来选择数组中对应的行
	bboxes = bboxes[filter_index]
	for bbox in bboxes:
		if bbox[2] >= 5 and bbox[3] >= 5 and bbox[4] <= w-5 and bbox[5] <= h-5:
			result.append([int(bbox[0]),bbox[1],int(bbox[2]),int(bbox[3]),int(bbox[4]),int(bbox[5])])
	
	return result

def letterbox_image(image, size):
    ih, iw = image.shape[:2]
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    new_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_image_padded = cv2.copyMakeBorder(new_image, 0, h - nh, 0, w - nw,
                                          cv2.BORDER_CONSTANT,
                                          value=[255, 255, 255])
    return new_image_padded


def getCropImg(img,bbox):
	'''
	get dete result
	'''

	batch_crop_img = []
	for bb in bbox:
		cls,conf,x1,y1,x2,y2 = bb
		h,w = y2-y1,x2-x1
		if h>=40 and w>=30:
			scale_h,scale_w = int(h/10),int(w/10)
			crop_img = img[y1-scale_h:y2+scale_h, x1-scale_w:x2+scale_w]
			crop_img = letterbox_image(crop_img, size=(224, 224))
			crop_img = crop_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
			np_img = np.ascontiguousarray(crop_img)
			np_img = np_img.astype(np.float32)  # float32
			np_img /= 255.0
			batch_crop_img.append(np_img)

	batch_crop_img = np.stack((batch_crop_img), axis=0)
	return batch_crop_img

def getClsResult(img,session):
    # 运行分类模型进行推理
    outputs = session.run([], {'input': img})
    result_list = []
    for i in range(outputs[0].shape[0]):
        single_list = []
        for output in outputs:
            output_sample = output[i]
            single_list.append(softmax(output_sample))
            #single_list.append(np.argmax(output_sample))
        result_list.append(single_list)
    return result_list
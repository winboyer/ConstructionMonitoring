import os
import time
from paddlex import create_model
from paddleocr import DocImgOrientationClassification

model = create_model(model_dir='/home/jinyfeng/models/PaddlePaddle/PP-LCNet_x1_0_doc_ori', 
                     model_name="PP-LCNet_x1_0_doc_ori")
                    # device='gpu:0')

# model = DocImgOrientationClassification(model_dir='/home/jinyfeng/models/PaddlePaddle/PP-LCNet_x1_0_doc_ori', 
#                                         model_name="PP-LCNet_x1_0_doc_ori")

data_path = '/home/jinyfeng/datas/data_test/'
# filename = '视频识别/管片识别/微信图片_20251106160954_213_226.jpg'
filename = '视频识别/管片识别/MVIMG_20251127_093229.jpg'
input_path = os.path.join(data_path, filename)

start_time = time.time()
output = model.predict(input=input_path, batch_size=1)
for res in output:
    print(res.get('label_names')[0])


# print("预测结果:", output,len(output))
# label_names = output[0].get('label_names', [])
# print(type(label_names), len(label_names),label_names, label_names[0])

end_time = time.time()
print(f"Inference time: {end_time - start_time:.2f} seconds")



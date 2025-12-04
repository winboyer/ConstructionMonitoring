from paddleocr import PaddleOCR
import time
import os

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

data_path = '/home/jinyfeng/datas/data_test/'

# Run OCR inference on a sample image 
start_time = time.time()
input_path = os.path.join(data_path, '6f22b81a-9b63-4465-aae6-5a2c41fdc8ca.png')
result = ocr.predict(input=input_path)
end_time = time.time()
print(f"Inference time: {end_time - start_time:.2f} seconds")

start_time = time.time()
input_path = os.path.join(data_path, '5295b714-b007-4490-994c-b3e3a7583f43.png')
result = ocr.predict(input=input_path)
end_time = time.time()
print(f"Inference time: {end_time - start_time:.2f} seconds")

start_time = time.time()
input_path = os.path.join(data_path, '893b5439-9d0b-44f8-9570-c0c06a22610e.png')
result = ocr.predict(input=input_path)
end_time = time.time()
print(f"Inference time: {end_time - start_time:.2f} seconds")

start_time = time.time()
input_path = os.path.join(data_path, '21abea49-4229-443f-8525-1de84d14a9b3.png')
result = ocr.predict(input=input_path)
end_time = time.time()
print(f"Inference time: {end_time - start_time:.2f} seconds")

start_time = time.time()
input_path = os.path.join(data_path, 'd4f40150-811c-4170-94b0-00df032f11fb.png')
result = ocr.predict(input=input_path)
end_time = time.time()
print(f"Inference time: {end_time - start_time:.2f} seconds")

# Visualize the results and save the JSON results
for res in result:
    # res.print()
    res.save_to_img("output.png")
    res.save_to_json("output.json")
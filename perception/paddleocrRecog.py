from paddleocr import PaddleOCR
import re
from datetime import datetime
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_process import get_image_dimensions
from PIL import Image
import numpy as np
import cv2

def is_valid_date_format(text):
    """
    检查字符串是否为'YYYY.MM.DD'格式的有效日期
    
    Args:
        text: 要检查的字符串
    
    Returns:
        bool: 是否为有效日期格式
        str: 错误信息（如果无效）
        datetime: 日期对象（如果有效）
    """
    try:
        # 尝试按照指定格式解析日期
        date_obj = datetime.strptime(text, '%Y.%m.%d')
        
        # 额外的验证：确保年份合理（例如2000-2030之间）
        if not 2000 <= date_obj.year <= 2030:
            return False, f"年份{date_obj.year}超出合理范围(2000-2030)", None
            
        return True, "日期格式正确且有效", date_obj
        
    except ValueError as e:
        # 检查常见错误类型
        if not text:
            return False, "字符串为空", None
        elif text.count('.') != 2:
            return False, "日期格式应为'YYYY.MM.DD'，包含两个点号", None
        else:
            return False, f"无效的日期: {str(e)}", None
    except Exception as e:
        return False, f"解析错误: {str(e)}", None

def get_image_file(img_file):
    if isinstance(img_file, str) and os.path.isfile(img_file):
        img = cv2.imread(img_file)
    elif isinstance(img_file, Image.Image):
        img = np.array(img_file)
    elif isinstance(img_file, bytes):
        from io import BytesIO
        img = Image.open(BytesIO(img_file))
        np_img = np.array(img)
        img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    elif isinstance(img_file, np.ndarray):
        if len(img_file.shape) == 3:
            img = img_file
        else:
            # 灰度图像
            img = cv2.cvtColor(img_file, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Unsupported image input type: {type(img_file)}")
    
    return img

def get_image_dimensions_resized(img_file, max_dimension=1296*1296):
    """Get image dimensions, resizing if necessary"""
    image_width, image_height = get_image_dimensions(img_file)
    print(f"Image size used for OCR: {image_width}x{image_height}")
    img = get_image_file(img_file)

    if image_width * image_height > max_dimension:
        scale_factor = (max_dimension / (image_width * image_height)) ** 0.5
        new_width = int(image_width * scale_factor)
        new_height = int(image_height * scale_factor)
        print(f"Resized image size: {new_width}x{new_height}")
        img = img.resize((new_width, new_height))
    else:
        new_width, new_height = image_width, image_height
    
    return new_width, new_height, img

class DocumentRecognizer:
    """Wrapper class for document recognition with OCR model"""
    
    def __init__(self, doc_orien_cls=False, doc_unwarping=False, textline_orien=False):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=doc_orien_cls,
            use_doc_unwarping=doc_unwarping,
            use_textline_orientation=textline_orien)
    
    def extract_info(self, img_path):
        """Extract document information from image"""
        # print(f"Extracting document info from image: {img_path}")
        _, _, image= get_image_dimensions_resized(img_path)
        result = self.ocr.predict(input=image)
        # print('prediction result length:', len(result))
        rec_texts = result[0].get('rec_texts', []) if result else []
        # print(len(rec_texts), rec_texts)
        
        return {
            'result': rec_texts,
            'total': len(rec_texts)
        }
    def extract_deliver_doc_info(self, img_path):
        """Extract delivery document information from image"""
        # print(f"Extracting delivery document info from image: {img_path}")
        image_width, image_height, image = get_image_dimensions_resized(img_path)
        print(f"Image size used for OCR: {image_width}x{image_height}")

        result = self.ocr.predict(input=image)
        print('prediction result length:', len(result), result)
        
        angle = result[0].get('angle', 0)
        print("angle", angle)
        # if angle == 90 or angle == 270:
        #     temp = image_width
        #     image_width = image_height
        #     image_height = temp
        
        print(f"Image size used for OCR: {image_width}x{image_height}")

        dt_polys = result[0].get('dt_polys', [])
        rec_polys = result[0].get('rec_polys', [])
        rec_texts = result[0].get('rec_texts', [])
        
        timestamp = ""
        project_name = ""
        product_name_num = 0
        product_name_list = []
        product_name_coords = []
        ring_id_list = []
        ring_id_coords = []
        product_cnt = 0
        product_cnt_list = []
        for idx, text in enumerate(rec_texts):
            if "轨道交通7号线" in text:
                project_name = text
                # print("施工项目：", text)
                # print(idx, rec_polys[idx])
            elif is_valid_date_format(text)[0]:
                timestamp = text
            elif "型" in text and "管片" in text:
                temp = text[:2]+'-'+text[-4:]
                # print('产品名称1', temp)
                # print(idx, rec_polys[idx])
                product_name_list.append(temp)
                center_x = (rec_polys[idx][0][0] + rec_polys[idx][1][0]+rec_polys[idx][2][0] + rec_polys[idx][3][0]) / 2
                center_y = (rec_polys[idx][0][1] + rec_polys[idx][1][1]+rec_polys[idx][2][1] + rec_polys[idx][3][1]) / 2
                product_name_coords.append((center_x, center_y))
            elif len(re.findall(r'-', text)) == 2:
                # print('环号：', text)
                # print(idx, rec_polys[idx])
                ring_id_list.append(text)
                center_x = (rec_polys[idx][0][0] + rec_polys[idx][1][0]+rec_polys[idx][2][0] + rec_polys[idx][3][0]) / 2
                center_y = (rec_polys[idx][0][1] + rec_polys[idx][1][1]+rec_polys[idx][2][1] + rec_polys[idx][3][1]) / 2
                ring_id_coords.append((center_x, center_y))
            elif bool(re.search(r'\.', text)) and bool(re.search(r'm', text)):
                print('产品名称2：', text)
                # print(idx, rec_polys[idx])
                center_x = (rec_polys[idx][0][0] + rec_polys[idx][1][0]+rec_polys[idx][2][0] + rec_polys[idx][3][0]) / 2
                center_y = (rec_polys[idx][0][1] + rec_polys[idx][1][1]+rec_polys[idx][2][1] + rec_polys[idx][3][1]) / 2
                print(center_x, image_width/2)
                if center_x < image_width / 2:
                    if not text[0]=="（" and not text[0]=="(":
                        text = "(" + text
                    if not text[-1]=="）" and not text[-1]==")":
                        text = text + ")"
                    product_name_list[product_name_num] = product_name_list[product_name_num] + text
                    product_name_num += 1
            elif text.isdigit() and int(text) < 10:
                center_x = (rec_polys[idx][0][0] + rec_polys[idx][1][0]+rec_polys[idx][2][0] + rec_polys[idx][3][0]) / 2
                center_y = (rec_polys[idx][0][1] + rec_polys[idx][1][1]+rec_polys[idx][2][1] + rec_polys[idx][3][1]) / 2
                # print('数量候选：', text)
                # print(len(ring_id_list), center_x)
                if len(product_cnt_list) < 1 and center_x > image_width / 2:
                    product_cnt = int(text)
                    product_cnt_list.append(product_cnt)
                    print("数量：", text)

        if len(product_cnt_list) < len(product_name_list):
            product_cnt_list.extend([product_cnt] * (len(product_name_list) - len(product_cnt_list)))

        return timestamp, project_name, product_name_list, ring_id_list, product_cnt_list

if __name__ == "__main__":
    data_path = '/Users/jinyfeng/个人文档/zhongjian_works/AI课题/施工进度估计/隧道施工项目/data_test/'
    import time
    start_time = time.time() 
    # input_path = os.path.join(data_path, '视频识别/管片识别/微信图片_20251106160954_213_226.jpg')
    input_path = os.path.join(data_path, '微信图片_20251216111447_251_2632.jpg')
    ocr_model = DocumentRecognizer(doc_orien_cls=True, doc_unwarping=False, textline_orien=False)
    image = cv2.imread(input_path)
    timestamp, project_name, product_name_list, ring_id_list, product_cnt_list = ocr_model.extract_deliver_doc_info(image)

    end_time = time.time()
    print("Total processing time:", end_time - start_time, "seconds")
    print("Timestamp:", timestamp)
    print("Project Name:", project_name)
    print("Product Names:", product_name_list)
    print("Ring IDs:", ring_id_list)
    print("Product Counts:", product_cnt_list)

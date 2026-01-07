"""
图像旋转工具模块
提供图像的90度、180度和270度旋转功能
"""

import os
import io
from PIL import Image
import base64
import numpy as np
import cv2


def rotate_image_90(image_input):
    """
    将图像顺时针旋转90度
    
    Args:
        image_input: 图像数据，可以是文件路径、base64字符串、bytes或numpy数组
        
    Returns:
        PIL.Image对象: 旋转后的图像
    """
    return _rotate_image(image_input, 90)


def rotate_image_180(image_input):
    """
    将图像顺时针旋转180度
    
    Args:
        image_input: 图像数据，可以是文件路径、base64字符串、bytes或numpy数组
        
    Returns:
        PIL.Image对象: 旋转后的图像
    """
    return _rotate_image(image_input, 180)


def rotate_image_270(image_input):
    """
    将图像顺时针旋转270度
    
    Args:
        image_input: 图像数据，可以是文件路径、base64字符串、bytes或numpy数组
        
    Returns:
        PIL.Image对象: 旋转后的图像
    """
    return _rotate_image(image_input, 270)


def _rotate_image(image_input, degrees):
    """
    内部函数：将图像旋转指定角度
    
    Args:
        image_input: 图像数据
        degrees (int): 旋转角度（90, 180, 或 270）
        
    Returns:
        PIL.Image对象: 旋转后的图像
    """
    # 处理不同类型的输入
    if isinstance(image_input, str) and os.path.isfile(image_input):
        # 文件路径
        image = Image.open(image_input)
    elif isinstance(image_input, str) and image_input.startswith('data:image/'):
        # base64数据（带data URL前缀）
        header, data = image_input.split(',', 1)
        image_bytes = base64.b64decode(data)
        image = Image.open(io.BytesIO(image_bytes))
    elif isinstance(image_input, str):
        # 纯base64字符串
        image_bytes = base64.b64decode(image_input)
        image = Image.open(io.BytesIO(image_bytes))
    elif isinstance(image_input, bytes):
        # bytes数据
        image = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, np.ndarray):
        # numpy数组（OpenCV格式）
        # 注意：OpenCV使用BGR格式，而PIL使用RGB格式
        if len(image_input.shape) == 3:
            # 如果是彩色图像，转换BGR到RGB
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            # 灰度图像
            image = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        # PIL Image对象
        image = image_input
    else:
        raise ValueError(f"不支持的图像输入类型: {type(image_input)}")
    
    # 执行旋转
    return image.rotate(degrees, expand=True)


def rotate_and_save_image(image_input, degrees, output_path):
    """
    旋转图像并保存到文件
    
    Args:
        image_input: 输入图像数据
        degrees (int): 旋转角度（90, 180, 或 270）
        output_path (str): 输出文件路径
        
    Returns:
        bool: 操作是否成功
    """
    try:
        rotated_image = _rotate_image(image_input, degrees)
        rotated_image.save(output_path)
        return True
    except Exception as e:
        print(f"旋转图像时出错: {e}")
        return False


def rotate_and_return_bytes(image_input, degrees):
    """
    旋转图像并返回bytes数据
    
    Args:
        image_input: 输入图像数据
        degrees (int): 旋转角度（90, 180, 或 270）
        
    Returns:
        bytes: 旋转后图像的bytes数据
    """
    try:
        rotated_image = _rotate_image(image_input, degrees)
        byte_io = io.BytesIO()
        rotated_image.save(byte_io, format='PNG')
        byte_io.seek(0)
        return byte_io.getvalue()
    except Exception as e:
        print(f"旋转图像时出错: {e}")
        return None


def rotate_and_return_base64(image_input, degrees):
    """
    旋转图像并返回base64编码字符串
    
    Args:
        image_input: 输入图像数据
        degrees (int): 旋转角度（90, 180, 或 270）
        
    Returns:
        str: 旋转后图像的base64编码字符串
    """
    try:
        rotated_image = _rotate_image(image_input, degrees)
        byte_io = io.BytesIO()
        rotated_image.save(byte_io, format='PNG')
        byte_io.seek(0)
        return base64.b64encode(byte_io.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"旋转图像时出错: {e}")
        return None


# 为了兼容性，添加一些便捷函数
def rotate_image(image_input, degrees):
    """
    通用图像旋转函数
    
    Args:
        image_input: 输入图像数据
        degrees (int): 旋转角度（90, 180, 或 270）
        
    Returns:
        PIL.Image对象: 旋转后的图像
    """
    if degrees == 90:
        return rotate_image_90(image_input)
    elif degrees == 180:
        return rotate_image_180(image_input)
    elif degrees == 270:
        return rotate_image_270(image_input)
    else:
        raise ValueError("仅支持90度、180度和270度旋转")


if __name__ == "__main__":
    # 测试示例
    import sys
    
    if len(sys.argv) != 3:
        print("用法: python image_rotate.py <input_image> <rotation_degrees>")
        print("支持的角度: 90, 180, 270")
        sys.exit(1)
    
    input_image = sys.argv[1]
    degrees = int(sys.argv[2])
    
    try:
        rotated = rotate_image(input_image, degrees)
        output_path = f"rotated_{degrees}deg.png"
        rotated.save(output_path)
        print(f"图像已旋转{degrees}度并保存为: {output_path}")
    except Exception as e:
        print(f"错误: {e}")

import os
import io
from PIL import Image
import base64
import numpy as np

def scale_person_bbox(bbox, img_width, img_height, scale=1.2):
    """
    Scale the person bounding box by a given factor.
    
    Args:
        bbox (list or tuple): Original bounding box [x_min, y_min, x_max, y_max]
        scale (float): Scaling factor
        img_width (int): Width of the image
        img_height (int): Height of the image
        
    Returns:
        list: Scaled bounding box [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = bbox
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    center_x = x_min + box_width / 2
    center_y = y_min + box_height / 2
    
    new_width = box_width * scale
    new_height = box_height * scale
    
    new_x_min = max(0, int(center_x - new_width / 2))
    new_y_min = max(0, int(center_y - new_height / 2))
    new_x_max = min(img_width, int(center_x + new_width / 2))
    new_y_max = min(img_height, int(center_y + new_height / 2))
    
    return [new_x_min, new_y_min, new_x_max, new_y_max]

def is_base64_image_data(image_input):
    """
    Check if the input is base64 encoded image data
    
    Args:
        image_input (str): Input to check
        
    Returns:
        bool: True if input appears to be base64 image data
    """
    # Check if it looks like base64 data (starts with data:image/)
    if isinstance(image_input, str) and image_input.startswith('data:image/'):
        return True
    
    # Check if it's valid base64 string
    try:
        if isinstance(image_input, str):
            # Remove data URL prefix if present
            if ',' in image_input:
                header, data = image_input.split(',', 1)
                if 'base64' in header:
                    # Try to decode base64
                    base64.b64decode(data, validate=True)
                    return True
            else:
                # Just try to decode as base64
                base64.b64decode(image_input, validate=True)
                return True
    except Exception:
        pass
    
    return False

def is_numpy_array(image_input):
    """
    Check if the input is a numpy array (OpenCV Mat data)
    
    Args:
        image_input: Input to check
        
    Returns:
        bool: True if input is a numpy array
    """
    return isinstance(image_input, np.ndarray)

def is_bytes_data(image_input):
    """
    Check if the input is bytes data
    
    Args:
        image_input: Input to check
        
    Returns:
        bool: True if input is bytes
    """
    return isinstance(image_input, bytes)

def get_image_dimensions_from_file(image_path):
    """
    Get the width and height of an image file
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (width, height) of the image, or (None, None) if error
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"Error opening image file {image_path}: {e}")
        return None, None

def get_image_dimensions_from_bytes(image_bytes):
    """
    Get the width and height of image bytes
    
    Args:
        image_bytes (bytes): Image data as bytes
        
    Returns:
        tuple: (width, height) of the image, or (None, None) if error
    """
    try:
        # Open image from bytes
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"Error processing image bytes: {e}")
        return None, None

def get_image_dimensions_from_numpy(image_array):
    """
    Get the width and height of OpenCV Mat data (numpy array)
    
    Args:
        image_array (numpy.ndarray): Image data as numpy array
        
    Returns:
        tuple: (width, height) of the image, or (None, None) if error
    """
    try:
        # OpenCV uses (height, width) format, PIL uses (width, height)
        if len(image_array.shape) == 3:
            height, width = image_array.shape[:2]
        elif len(image_array.shape) == 2:
            height, width = image_array.shape
        else:
            raise ValueError("Unsupported image array shape")
        return width, height
    except Exception as e:
        print(f"Error processing numpy array: {e}")
        return None, None

def get_image_dimensions_from_data(image_data):
    """
    Get the width and height of image data (base64, bytes, or numpy array)
    
    Args:
        image_data: Image data in various formats
        
    Returns:
        tuple: (width, height) of the image, or (None, None) if error
    """
    # Handle base64 data
    if is_base64_image_data(image_data):
        try:
            # Handle data URL format (data:image/png;base64,...)
            if isinstance(image_data, str) and image_data.startswith('data:image/'):
                header, data = image_data.split(',', 1)
                image_bytes = base64.b64decode(data)
            elif isinstance(image_data, str):
                # Handle plain base64 string
                image_bytes = base64.b64decode(image_data)
            else:
                # Assume it's already bytes
                image_bytes = image_data
                
            return get_image_dimensions_from_bytes(image_bytes)
        except Exception as e:
            print(f"Error decoding base64 data: {e}")
            return None, None
    
    # Handle bytes data
    elif is_bytes_data(image_data):
        return get_image_dimensions_from_bytes(image_data)
    
    # Handle numpy array (OpenCV Mat)
    elif is_numpy_array(image_data):
        return get_image_dimensions_from_numpy(image_data)
    
    elif isinstance(image_data, Image.Image):
        return image_data.size  # (width, height)
    # If none of the above, return None
    else:
        print(f"Unknown input type: {type(image_data)}")
        return None, None

def get_image_dimensions(image_input):
    """
    Get the width and height of an image (file path, bytes, or numpy array)
    
    Args:
        image_input: Path to image file, base64 data, bytes, or numpy array
        
    Returns:
        tuple: (width, height) of the image, or (None, None) if error
    """
    # Check if input is a file path
    if isinstance(image_input, str) and os.path.isfile(image_input):
        return get_image_dimensions_from_file(image_input)
    else:
        # Treat as data (base64, bytes, or numpy array)
        return get_image_dimensions_from_data(image_input)
    

def bytes_to_base64(image_bytes):
    """
    Convert image bytes to base64 encoded string.
    
    Args:
        image_bytes: Image data in bytes format
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_bytes).decode('utf-8')

def base64_to_bytes(base64_string):
    """
    Decode base64 encoded string to image bytes.
    
    Args:
        base64_string: Base64 encoded string
        
    Returns:
        Image data in bytes format
    """
    return base64.b64decode(base64_string)


def mask_to_bbox(mask: np.ndarray):
    """根据二值掩膜计算最小外接矩形"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None  # 没有目标
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


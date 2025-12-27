import base64

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



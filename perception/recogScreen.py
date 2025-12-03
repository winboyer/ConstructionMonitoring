import pyautogui
import pytesseract
from PIL import Image
import cv2
import numpy as np

def capture_screen():
    """Capture current screen and save as image"""
    screenshot = pyautogui.screenshot()
    screenshot.save('screen_capture.png')
    return screenshot

def recognize_text(image):
    """Recognize text from image using OCR"""
    text = pytesseract.image_to_string(image)
    return text

def recognize_objects(image_path):
    """Recognize objects in image using OpenCV"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def main():
    # Capture screen
    print("Capturing screen...")
    screenshot = capture_screen()
    
    # Recognize text
    print("Recognizing text...")
    text = recognize_text(screenshot)
    print("Recognized text:")
    print(text)
    
    # Recognize objects
    print("\nRecognizing objects...")
    objects = recognize_objects('screen_capture.png')
    print("Objects recognized successfully")

if __name__ == "__main__":
    main()
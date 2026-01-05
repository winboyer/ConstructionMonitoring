from paddleocr import PaddleOCR

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
        result = self.ocr.predict(input=img_path)
        # print('prediction result length:', len(result))
        rec_texts = result[0].get('rec_texts', []) if result else []
        # print(len(rec_texts), rec_texts)
        
        return {
            'result': rec_texts,
            'total': len(rec_texts)
        }



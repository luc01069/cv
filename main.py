import os
import numpy as np
from vision_agent.tools import *
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool


from vision_agent.tools import load_image, florence2_sam2_image, ocr, qwen2_vl_images_vqa
import numpy as np

def analyze_mechanical_component(image_path):
    # Load the image
    image = load_image(image_path)
    
    # Use florence2_sam2_image to segment cylindrical parts
    segmentation_results = florence2_sam2_image('cylindrical parts, shaft sections', image)
    
    # Use ocr to extract text and dimensions
    ocr_results = ocr(image)
    
    # Use qwen2_vl_images_vqa to interpret results and describe parts
    prompt = "Describe in detail the different cylindrical sections of this mechanical component, their dimensions, and any other relevant features. Use the segmentation and OCR results to provide accurate information."
    interpretation = qwen2_vl_images_vqa(prompt, [image])
    
    # Process and combine results
    height, width = image.shape[:2]
    
    processed_segments = []
    for segment in segmentation_results:
        x1, y1, x2, y2 = segment['bbox']
        processed_segments.append({
            'label': segment['label'],
            'bbox': [
                int(x1 * width),
                int(y1 * height),
                int(x2 * width),
                int(y2 * height)
            ],
            'score': segment['score']
        })
    
    processed_ocr = []
    for text in ocr_results:
        x1, y1, x2, y2 = text['bbox']
        processed_ocr.append({
            'text': text['label'],
            'bbox': [
                int(x1 * width),
                int(y1 * height),
                int(x2 * width),
                int(y2 * height)
            ],
            'score': text['score']
        })
    
    # Return a dictionary with segmentation, dimensions, and description
    return {
        'segmentation': processed_segments,
        'ocr_results': processed_ocr,
        'interpretation': interpretation
    }


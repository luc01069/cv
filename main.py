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
    """
    Analyzes a mechanical component by loading the image, segmenting cylindrical parts,
    extracting text and dimensions using OCR, and interpreting the results to describe the parts.

    Args:
        image_path (str): The file path to the image of the mechanical component.

    Returns:
        dict: A dictionary containing the interpretation of the mechanical component.
    """
    # Load the image
    image = load_image(image_path)
    
    # Use florence2_sam2_image to segment cylindrical parts
    segmentation_results = florence2_sam2_image('cylindrical parts, shaft sections', image)
    
    # Use ocr to extract text and dimensions
    ocr_results = ocr(image)
    
    # Use qwen2_vl_images_vqa to interpret results and describe parts
    prompt = "Describe in detail the different cylindrical sections of this mechanical component, their dimensions, and any other relevant features. Use the segmentation and OCR results to provide accurate information."
    interpretation = qwen2_vl_images_vqa(prompt, [image])

    return interpretation

def process_and_combine_results(image, segmentation_results, ocr_results):
    """
    Processes and combines segmentation and OCR results by scaling bounding box coordinates to the image size.

    Args:
        image (numpy.ndarray): The input image.
        segmentation_results (list): A list of segmentation results, where each result is a dictionary containing:
            - 'label' (str): The label of the segment.
            - 'bbox' (list): The bounding box coordinates [x1, y1, x2, y2].
            - 'score' (float): The confidence score of the segment.
        ocr_results (list): A list of OCR results, where each result is a dictionary containing:
            - 'label' (str): The recognized text.
            - 'bbox' (list): The bounding box coordinates [x1, y1, x2, y2].

    Returns:
        tuple: A tuple containing:
            - processed_segments (list): A list of processed segmentation results with scaled bounding boxes.
            - processed_ocr (list): A list of processed OCR results with scaled bounding boxes.
    """
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
            ]
        })
    
    return processed_segments, processed_ocr

def main():
    """
    Main function to analyze a mechanical component image.
    """
    image_path = 'path/to/your/image.jpg'  # Replace with the actual image path
    interpretation = analyze_mechanical_component(image_path)
    print(interpretation)

if __name__ == "__main__":
    main()
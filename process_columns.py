import cv2
import numpy as np
import json
import os

def process_columns(image_path, output_image_path, output_json_path):
    """
    Detects columns in an image, annotates them, and saves the results.

    Args:
        image_path (str): Path to the input image.
        output_image_path (str): Path to save the annotated image.
        output_json_path (str): Path to save the column boundary data in JSON format.
    """
    # Validate input image path
    if not os.path.exists(image_path):
        print(f"Error: Input image '{image_path}' does not exist.")
        return

    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 100]  


    column_data = []
    for i, contour in enumerate(contours):
        color = tuple(np.random.randint(0, 256, size=3).tolist())  # Generate random color
        cv2.drawContours(image, [contour], -1, color, 2)
        column_data.append({
            "column_id": i + 1,
            "boundary_points": contour.reshape(-1, 2).tolist()
        })

    try:
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        cv2.imwrite(output_image_path, image)
        with open(output_json_path, 'w') as f:
            json.dump(column_data, f, indent=4)
        print(f"Processing complete!")
        print(f"Annotated image saved to: {output_image_path}")
        print(f"Column boundary data saved to: {output_json_path}")
    except Exception as e:
        print(f"Error saving output: {e}")

if __name__ == "__main__":
    input_image_path = "input_image.jpg"
    output_image_path = "output/output_image.jpg"
    output_json_path = "output/columns.json"

    process_columns(input_image_path, output_image_path, output_json_path)

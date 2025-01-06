import cv2
import numpy as np
import json
import os

def process_columns(image_path, output_image_path, output_json_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (128, 128, 0), (0, 128, 128), (128, 0, 128)
    ]

    # Initialize data to dump
    column_data = []

    # Draw and process each column
    for i, contour in enumerate(contours):
        color = colors[i % len(colors)]

        cv2.drawContours(image, [contour], -1, color, 2)

        points = contour.reshape(-1, 2).tolist()
        column_data.append({"column_id": i + 1, "boundary_points": points})

    # Output the image
    cv2.imwrite(output_image_path, image)

    with open(output_json_path, 'w') as f:
        json.dump(column_data, f, indent=4)

    print(f"Processing complete!")
    print(f"Annotated image saved to: {output_image_path}")
    print(f"Column boundary data saved to: {output_json_path}")

if __name__ == "__main__":
    input_image_path = "input_image.jpg"
    output_image_path = "output_image.jpg"
    output_json_path = "columns.json"

    if not os.path.exists(input_image_path):
        print(f"Error: Input image {input_image_path} does not exist.")
    else:
        process_columns(input_image_path, output_image_path, output_json_path)

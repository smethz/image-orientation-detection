import os
from PIL import Image
import random
import csv

csv_file = 'data_rotation.csv'
input_dir = './original'
output_dir = './rotated'
image_extension = '.webp'

def rotate_image_random(image, output_path):
    # List of angles divisible by 30 degrees
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    # Choose a random angle from the list
    angle = random.choice(angles)
    
    # Rotate the image
    rotated_image = image.rotate(angle, expand=True)
    
    # Save the rotated image
    rotated_image.save(output_path)
    
    return angle

file_exists = os.path.exists(csv_file)
file_is_empty = os.stat(csv_file).st_size == 0 if file_exists else True

def process_images_in_directory(input_directory, output_directory):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
    
        # Write headers only if the file is empty
        if file_is_empty:
            writer.writerow(['path', 'degree'])

        # List all files in the input directory
        for filename in os.listdir(input_directory):
            file_path = os.path.join(input_directory, filename)
        
            # Check if the file is an image
            if os.path.isfile(file_path) and filename.lower().endswith((image_extension)):
                try:
                    # Open the image
                    image = Image.open(file_path)
                
                    # Define output path
                    output_path = os.path.join(output_directory, filename)
                
                    # Rotate and save the image
                    angle = rotate_image_random(image, output_path)
                
                    print(f"Image rotated by {angle} degrees and saved to {output_path}")

                    angle = (360 - angle) % 360
                    data = [f'{output_path}', f'{angle}']
                    writer.writerow(data)
                except Exception as e:
                    print(f"Could not process {filename}: {e}")

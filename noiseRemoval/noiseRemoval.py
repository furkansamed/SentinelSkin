import os
import cv2
import numpy as np

# Convert a color image to grayscale
def convert_to_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Apply Morphological Black-Hat transformation on a grayscale image
def apply_blackhat_transform(image):
    kernel = np.ones((15, 15), np.uint8)
    blackhat_image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    return blackhat_image

# Create a mask for InPainting operation
def create_mask(image):
    _, thresh = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(thresh, None, iterations=2)
    return mask

# Apply InPainting algorithm on the original image
def apply_inpainting(image, mask):
    inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return inpainted_image

# Input folders
input_folders = [
    r'C:\\Users\\Monster\\Desktop\\proje\\main\\data\\data_raw\\test_benign_raw',
    r'C:\\Users\\Monster\\Desktop\\proje\\main\\data\\data_raw\\test_malignant_raw',
    r'C:\\Users\\Monster\\Desktop\\proje\\main\\data\\data_raw\\train_benign_raw',
    r'C:\\Users\\Monster\\Desktop\\proje\\main\\data\\data_raw\\train_malignant_raw'
]

# Output folders
output_folders = [
    r'C:\\Users\\Monster\\Desktop\\proje\\main\\data\\data_denoised\\test_benign_denoised',
    r'C:\\Users\\Monster\\Desktop\\proje\\main\\data\\data_denoised\\test_malignant_denoised',
    r'C:\\Users\\Monster\\Desktop\\proje\\main\\data\\data_denoised\\train_benign_denoised',
    r'C:\\Users\\Monster\\Desktop\\proje\\main\\data\\data_denoised\\train_malignant_denoised'
]

# Process files in a loop
for input_folder, output_folder in zip(input_folders, output_folders):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Process only files with specific extensions
            # Load the image
            image_path = os.path.join(input_folder, filename)
            original_image = cv2.imread(image_path)

            # Apply the steps
            grayscale_image = convert_to_grayscale(original_image)
            blackhat_image = apply_blackhat_transform(grayscale_image)
            mask = create_mask(blackhat_image)
            inpainted_image = apply_inpainting(original_image, mask)

            # Save the inpainted image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, inpainted_image)

print("Denoising operation is finished")
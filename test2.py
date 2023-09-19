# import cv2

# # Read the images
# foreground = cv2.imread("./img/graphics/0 Pembelton.png", cv2.IMREAD_UNCHANGED)
# background = cv2.imread("./img/1.jpg", cv2.IMREAD_COLOR)

import cv2
import numpy as np

# Load the background image
background = cv2.imread("./img/1.jpg", cv2.IMREAD_COLOR)

# Load the PNG image with transparency
png_image = cv2.imread("./img/graphics/0 Pembelton.png", cv2.IMREAD_UNCHANGED)

scale = 6
png_image = cv2.resize(png_image, (png_image.shape[1] // 6, png_image.shape[0] // 6))

# Extract the alpha channel from the PNG image
alpha_channel = png_image[:, :, 3]

# Find the dimensions of both images

png_height, png_width = png_image.shape[:2]

# Calculate the position where the PNG image should be placed on the background
x_position = 200  # Adjust this value to position the PNG image as needed
y_position = 100  # Adjust this value to position the PNG image as needed

# Create a region of interest (ROI) for placing the PNG image
roi = background[
    y_position : y_position + png_height, x_position : x_position + png_width
]

# Use the alpha channel to blend the PNG image onto the background
for c in range(0, 3):
    background[
        y_position : y_position + png_height, x_position : x_position + png_width, c
    ] = background[
        y_position : y_position + png_height, x_position : x_position + png_width, c
    ] * (
        1 - alpha_channel / 255.0
    ) + png_image[
        :, :, c
    ] * (
        alpha_channel / 255.0
    )

# Display the result
cv2.imshow("Result", background)
cv2.waitKey(0)
cv2.destroyAllWindows()

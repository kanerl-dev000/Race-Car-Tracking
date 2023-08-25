import cv2
import numpy as np

# Load the main image and the title box image with transparency
main_img = cv2.imread("./img/1.jpeg")
overlay_img = cv2.imread(
    "./img/background.png", cv2.IMREAD_UNCHANGED
)  # Load with alpha channel

print(overlay_img.shape[1])

overlay_img = cv2.resize(
    overlay_img, (overlay_img.shape[1] // 4, overlay_img.shape[0] // 4)
)
# Ensure the main image is in 3 channel format
if main_img.shape[2] == 1:
    main_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR)

# # Resize the overlay image to the desired dimensions
# desired_width, desired_height = 100, 50
# overlay_img = cv2.resize(overlay_img, (desired_width, desired_height))

# Extract the alpha channel from the overlay image and normalize it
alpha = overlay_img[:, :, 3] / 255.0
alpha_inv = 1.0 - alpha

# Define the region where you want to place the overlay image
y1, y2 = 10, 10 + overlay_img.shape[0]
x1, x2 = 10, 10 + overlay_img.shape[1]

# For regions of interest in both images
for c in range(0, 3):
    main_img[y1:y2, x1:x2, c] = (
        alpha * overlay_img[:, :, c] + alpha_inv * main_img[y1:y2, x1:x2, c]
    )

# Save or display the result
cv2.imwrite("result_image.jpg", main_img)
cv2.imshow("Overlayed Image", main_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

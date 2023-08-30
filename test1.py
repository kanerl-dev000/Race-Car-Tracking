import cv2
import numpy as np

from PIL import ImageFont, ImageDraw, Image

image = np.zeros((500, 500, 3), dtype=np.uint8)


font_path = "./Font/AgencyFBBold.ttf"


font = ImageFont.truetype(font_path, 62)

img_pil = Image.fromarray(image)
draw = ImageDraw.Draw(img_pil)
b, g, r, a = 0, 255, 0, 0
draw.text((50, 100), "Hello World", font=font, fill=(b, g, r, a))

image = np.array(img_pil)

# text = "Hello, World!"
# font_size = 30
# thickness = 2
# text_width, text_height = font.gettextsize(text, font_size, thickness)

# Position of the text
# x = int((image.shape[1] - text_width) / 2)
# y = int((image.shape[0] + text_height) / 2)

# Draw the text on the image
# image = cv2.putText(
#     image,
#     text,
#     (x, y),
#     font_size,
#     thickness,
#     color=(255, 255, 255),
#     line_type=cv2.LINE_AA,
#     bottomLeftOrigin=True,
# )

# Display the resulting image
cv2.imshow("res", image)
cv2.waitKey()
cv2.destroyAllWindows()

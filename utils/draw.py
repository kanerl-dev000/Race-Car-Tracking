import numpy as np
import cv2

import time

from PIL import ImageFont, ImageDraw, Image


def puttitlebox(
    img,
    bg_width,
    x_margin,
    y_margin,
    space,
    box,
    index,
    identities,
    titles,
    font_path_num,
    font_path_drive,
):
    label_num = ""
    label_driver = ""
    currentX = 0

    index_of_title = next(
        (
            idx
            for idx, title in enumerate(titles)
            if title["trackid"] == identities[index]
        ),
        None,
    )
    try:
        label_num = str(titles[index_of_title]["number"])
        label_driver = titles[index_of_title]["name"]
        currentX = titles[index_of_title]["currentX"]
    except:
        label_num = "01"
        label_driver = "Driver"

    x1, y1, w1, h1, box_id = [int(i) for i in box]
    x2 = x1 + w1 // 2
    y2 = y1 + h1 // 2
    x1 = x1 - w1 // 2
    y1 = y1 - h1 // 2

    # Set title box coordinate
    pdsx = x_margin + index * (bg_width + space)
    pdsy = y_margin

    step = 0
    targetx = x_margin + index * (bg_width + space)
    dist = targetx - currentX

    if abs(dist) < 12:
        if abs(dist) < 5:
            step = dist
        else:
            step = 4 * np.sign(dist)
    else:
        step = (dist) // 3

    pdsx = currentX + step
    currentX = pdsx
    titles[index_of_title]["currentX"] = currentX

    # Define the region where you want to place the overlay image
    yy1 = pdsy
    xx1, xx2 = pdsx, pdsx + bg_width

    px0 = (x1 + x2) // 2
    py0 = (y1 + y2) // 2
    px1 = (xx2 + xx1) // 2 - 5
    py1 = yy1 + 50
    px2 = px1 + 10

    starty = py1
    endy = py0

    startx = px1
    middlex = px0
    endx = px2

    half = (endy - starty) // 2 + starty
    for y in range(starty, endy + 1):
        if endy - starty == 0:
            t = 0
        else:
            t = (y - starty) / (endy - starty)  # Calculate the ratio for gradient

        if y < half:
            color = (
                int(t * 170 + 80),
                int(t * 170 + 80),
                int(t * 170 + 80),
            )
        else:
            color = (
                int((1 - t) * 170 + 80),
                int((1 - t) * 170 + 80),
                int((1 - t) * 170 + 80),
            )
        cv2.line(
            img,
            (middlex + int((startx - middlex) * (1 - t)), y),
            (middlex + int((endx - middlex) * (1 - t)), y),
            color,
            1,
        )

    font_size1 = 60
    font_size2 = 50

    font1 = ImageFont.truetype(font_path_num, font_size1)
    font2 = ImageFont.truetype(font_path_drive, font_size2)

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    b, g, r, a = 255, 255, 255, 0

    t_size_num = (int(draw.textlength(label_num, font1)), font_size1)
    t_size_driver = (int(draw.textlength(label_driver, font2)), font_size2)

    text_length = t_size_num[0] + t_size_driver[0] + 20

    # draw.text(
    #     (
    #         pdsx + (bg_width - text_length) // 2,
    #         pdsy - 5,
    #     ),
    #     label_num,
    #     font=font1,
    #     fill=(b, g, r, a),
    # )

    draw.text(
        (
            pdsx + (bg_width - text_length) // 2 + t_size_num[0] + 20,
            pdsy,
        ),
        label_driver,
        font=font2,
        fill=(b, g, r, a),
    )

    img = np.array(img_pil)

    g_positionX = pdsx + (bg_width - text_length) // 2 - 25
    g_positionY = pdsy + 5

    png_image = cv2.imread(
        f"./img/graphics/{titles[index_of_title]['gnName']}", cv2.IMREAD_UNCHANGED
    )

    img = put_graphics(
        background=img,
        png_image=png_image,
        position=[g_positionX, g_positionY],
    )

    return img, titles


def put_graphics(background, png_image, position):
    scale = 4
    png_image = cv2.resize(
        png_image, (png_image.shape[1] // scale, png_image.shape[0] // scale)
    )

    # Extract the alpha channel from the PNG image
    alpha_channel = png_image[:, :, 3]

    # Find the dimensions of both images
    png_height, png_width = png_image.shape[:2]

    x_position, y_position = position
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

    return background


def draw_boxes(
    img,
    bbox,
    identities=None,
    titles={},
    font_path_num=None,
    font_path_drive=None,
):
    n_cars = len(identities)  # number of titles

    # Ensure the main image is in 3 channel format
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    bg_width = 200

    # spcae between titles
    space = (img.shape[1] - bg_width * n_cars) // (n_cars + 1)

    # Margin of title
    x_margin = space
    y_margin = img.shape[0] // 6

    # Sort titles
    identities_reshape = np.array(identities).reshape(-1, 1)
    arr = np.hstack((bbox, identities_reshape))
    sorted_indices = np.argsort(arr[:, 0])
    sorted_arr = arr[sorted_indices]
    identities = []

    for x in list(sorted_arr):
        identities.append(x[4])

    for i, box in enumerate(sorted_arr):
        img, titles = puttitlebox(
            img,
            bg_width,
            x_margin,
            y_margin,
            space,
            box,
            i,
            identities,
            titles,
            font_path_num,
            font_path_drive,
        )
    print(titles)
    return img, titles

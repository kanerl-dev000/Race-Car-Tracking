import numpy as np
import cv2

import time

palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def puttitlebox(
    img,
    bg_im,
    bg_width,
    bg_height,
    x_margin,
    y_margin,
    space,
    box,
    index,
    identities,
    titles,
):
    x1, y1, w1, h1, box_id = [int(i) for i in box]
    x2 = x1 + w1 // 2
    y2 = y1 + h1 // 2
    x1 = x1 - w1 // 2
    y1 = y1 - h1 // 2

    # Set title box coordinate
    pdsx = x_margin + index * (bg_width + space)
    pdsy = y_margin
    # pdex = pdsx + bg_width
    # pdey = bg_height + pdsy

    # Extract the alpha channel from the bg_im and normalize it
    alpha = bg_im[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha

    # Define the region where you want to place the overlay image
    yy1, yy2 = pdsy, pdsy + bg_height
    xx1, xx2 = pdsx, pdsx + bg_width
    # For regions of interest in both images
    for c in range(0, 3):
        img[yy1:yy2, xx1:xx2, c] = (
            alpha * bg_im[:, :, c] + alpha_inv * img[yy1:yy2, xx1:xx2, c]
        )

    px0 = (x1 + x2) // 2
    py0 = (y1 + y2) // 2
    px1 = (xx2 + xx1) // 2 - 5
    py1 = yy2
    px2 = px1 + 10
    # py2 = py1

    # points = [
    #     (px0, py0),
    #     (px2, py1),
    #     (px1, py2),
    # ]

    starty = py1
    endy = py0

    startx = px1
    middlex = px0
    endx = px2

    half = (endy - starty) // 2 + starty
    for y in range(starty, endy + 1):
        t = (y - starty) / (endy - starty)  # Calculate the ratio for gradient

        # print(t)
        if y < half:
            color = (
                int(t * 200 + 50),
                int(t * 200 + 50),
                int(t * 200 + 50),
            )
        else:
            color = (
                int((1 - t) * 200 + 50),
                int((1 - t) * 200 + 50),
                int((1 - t) * 200 + 50),
            )
        cv2.line(
            img,
            (middlex + int((startx - middlex) * (1 - t)), y),
            (middlex + int((endx - middlex) * (1 - t)), y),
            color,
            1,
        )

    # img = cv2.fillPoly(img, [np.array(points)], (150, 149, 146))

    label_num = ""
    label_driver = ""

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
    except:
        label_num = "01"
        label_driver = "Driver"

    t_size_num = cv2.getTextSize(label_num, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    t_size_driver = cv2.getTextSize(label_driver, cv2.FONT_HERSHEY_PLAIN, 1.5, 2)[0]

    cv2.putText(
        img,
        label_num,
        (
            pdsx + bg_width // 8 - t_size_num[0] // 2,
            pdsy + bg_height // 2 + t_size_num[1] // 2,
        ),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        [255, 255, 255],
        2,
    )
    cv2.putText(
        img,
        label_driver,
        (
            pdsx + bg_width * 5 // 8 - t_size_driver[0] // 2,
            pdsy + bg_height // 2 + t_size_driver[1] // 2,
        ),
        cv2.FONT_HERSHEY_PLAIN,
        1.5,
        [255, 255, 255],
        2,
    )
    return img


def draw_boxes(img, bbox, identities=None, bg_im=None, titles={}):
    start = time.time()
    n_cars = len(identities)  # number of titles

    # Ensure the main image is in 3 channel format
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Resize bg_im as 1/scale
    bg_scale = 4.5
    bg_width = int(bg_im.shape[1] / bg_scale)
    bg_height = int(bg_im.shape[0] / bg_scale)
    bg_im = cv2.resize(bg_im, (bg_width, bg_height))

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
        img = puttitlebox(
            img,
            bg_im,
            bg_width,
            bg_height,
            x_margin,
            y_margin,
            space,
            box,
            i,
            identities,
            titles,
        )
    return img


if __name__ == "__main__":
    for i in range(82):
        print(compute_color_for_labels(i))

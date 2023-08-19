import numpy as np
import cv2

palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, bg_im=None, titles={}, scale=1):
    n_cars = len(identities)
    height, width, channels = img.shape
    bg_width = width * 8 // 10 // n_cars
    if bg_width > 600:
        bg_width = 600
    bg_height = height // 15
    bg_im = cv2.resize(bg_im, (bg_width, bg_height))
    space = width // 10 // n_cars
    x_margin = width // 10
    y_margin = height // 15

    identities_reshape = np.array(identities).reshape(-1, 1)
    arr = np.hstack((bbox, identities_reshape))
    sorted_indices = np.argsort(arr[:, 0])
    sorted_arr = arr[sorted_indices]
    identities = []

    for x in list(sorted_arr):
        identities.append(x[4])

    for i, box in enumerate(sorted_arr):
        # x1, y1, x2, y2, box_id = [int(i) for i in box]
        # w = x2 - x1
        # h = y2 - y1
        # x1 = x1 + w // 2
        # x2 = x2 + w // 2
        # y1 = y1 + h // 2
        # y2 = y2 + h // 2

        x1, y1, w1, h1, box_id = [int(i) for i in box]

        x1 = int(scale * x1)
        y1 = int(scale * y1)
        w1 = int(scale * w1)
        h1 = int(scale * h1)

        x2 = x1 + w1 // 2
        y2 = y1 + h1 // 2
        x1 = x1 - w1 // 2
        y1 = y1 - h1 // 2

        # # put desciption part
        thickness = 2
        d_color = (145, 158, 144)
        pdsx = x_margin + i * (bg_width + space)
        pdsy = y_margin
        pdex = pdsx + bg_width
        pdey = bg_height + pdsy

        img[pdsy:pdey, pdsx:pdex] = bg_im
        pox = (x1 + x2) // 2
        poy = y1
        pdx = (pdex + pdsx) // 2
        pdy = pdey
        points = [
            ((x1 + x2) // 2, (y1 + y2) // 2),
            ((pdex + pdsx) // 2 + 5, pdey),
            ((pdex + pdsx) // 2 - 5, pdey),
        ]

        img = cv2.fillPoly(img, [np.array(points)], (204, 207, 205))
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = ""

        try:
            label = str(titles[i]["number"]) + " " + titles[i]["name"]
        except:
            label = "001 Speed"

        # if len(title) <= i:
        #     label = "1 Speed"
        # else:
        #     label = str(title[i]["number"]) + " " + title[i]["name"]
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]

        cv2.putText(
            img,
            label,
            (
                pdsx + bg_width // 2 - t_size[0] // 2,
                pdsy + bg_height // 2 + t_size[1] // 2,
            ),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            [255, 255, 255],
            2,
        )
        # cv2.putText(
        #     img,
        #     title["name"],
        #     (pdsx + label_lm, pdsy),
        #     cv2.FONT_HERSHEY_PLAIN,
        #     2,
        #     [255, 255, 255],
        #     2,
        # )
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        # cv2.rectangle(
        #     img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1
        # )
        # cv2.putText(
        #     img,
        #     label,
        #     (x1, y1 + t_size[1] + 4),
        #     cv2.FONT_HERSHEY_PLAIN,
        #     2,
        #     [255, 255, 255],
        #     2,
        # )
        # cv2.putText(img,names[i],(x1+20,y1+t_size[1]+20), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img


if __name__ == "__main__":
    for i in range(82):
        print(compute_color_for_labels(i))

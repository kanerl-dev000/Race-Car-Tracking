import cv2
from ultralytics import YOLO

import numpy as np

from utils.draw import draw_boxes

import time


class CarTrack(object):
    def __init__(self):
        # Load the YOLOv8 model
        self.model = YOLO("./weights/best.pt")
        # Open the video file
        self.mouse = [-1000, -1000]

        self.isMouseOver = False
        self.isClick = True

        self.track_ids = []

        self.targetID = []

        self.titles = []

        self.bg_im = cv2.imread("./img/background.png", cv2.IMREAD_UNCHANGED)

        self.ids = "hello"

        self.carids = []

    def __enter__(self):
        pass
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def xywh_to_xyxy(self, box):
        xx, yy, ww, hh = box
        return [xx - ww // 2, yy - hh // 2, xx + ww // 2, yy + hh // 2]

    def run(self, frame=None):
        start = time.time()
        # Loop through the video frames
        img = frame
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = self.model.track(img, persist=True)
        print("Inference time:      ", time.time() - start)
        boxes = results[0].boxes.xywh.cpu()
        if not results[0].boxes.id == None:
            self.track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            return frame, self.isMouseOver
        ## Add target cars
        if self.isClick:
            self.bufferID = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = self.xywh_to_xyxy(box)

                if self.mouse[0] in range(int(x1), int(x2)) and self.mouse[1] in range(
                    int(y1), int(y2)
                ):
                    id = self.track_ids[i]
                    if id not in self.targetID:
                        if (
                            len(self.targetID) >= len(self.titles)
                            or len(self.titles) == 0
                        ):
                            print("Select title")
                        else:
                            self.targetID.append(id)
                            self.titles[len(self.targetID) - 1]["trackid"] = id

                    else:
                        self.targetID.remove(id)
                        index_to_remove = next(
                            (
                                index
                                for index, obj in enumerate(self.titles)
                                if obj["trackid"] == id
                            ),
                            None,
                        )
                        if index_to_remove is not None:
                            self.titles.pop(index_to_remove)
                            self.carids.pop(index_to_remove)

                        self.isClick = False

        ## Check if there is target cars in boxes list
        j = 0
        while j < len(self.targetID):
            if self.targetID[j] not in self.track_ids:
                self.targetID.pop(j)
            else:
                j += 1
        ## Set Mouse State
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = self.xywh_to_xyxy(box)
            if self.mouse[0] in range(int(x1), int(x2)) and self.mouse[1] in range(
                int(y1), int(y2)
            ):
                self.isMouseOver = True
                break
            elif i == len(boxes) - 1:
                self.isMouseOver = False

        ## Draw title
        targetCars = []
        for id in self.targetID:
            targetCars.append(boxes[self.track_ids.index(id)])
        img0 = frame
        print("Track time before draw:      ", time.time() - start)
        if len(self.targetID) > 0:
            img0 = draw_boxes(
                img=img0,
                bbox=np.array(targetCars),
                identities=self.targetID,
                bg_im=self.bg_im,
                titles=self.titles,
            )
        print("Track time:      ", time.time() - start)
        return img0, self.isMouseOver

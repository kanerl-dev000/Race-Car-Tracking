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
        self.bufferID = []
        self.bufferBOX = []

        self.targetID = []

        self.titles = []

        self.bg_im = cv2.imread("./img/background.PNG")

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

    def chooseOneID(self):
        minID = 0
        minDistance = 100000
        for bufferID in self.bufferID:
            x, y, w, h = bufferID[1]
            dist = (self.mouse[0] - x) ** 2 + (self.mouse[1] - y) ** 2
            if minDistance > dist:
                minDistance = dist
                minID = bufferID[0]
        return minID

    def run(self, frame=None):
        # Loop through the video frames
        img = frame
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = self.model.track(img, persist=True)
        boxes = results[0].boxes.xywh.cpu()
        if not results[0].boxes.id == None:
            self.track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            return frame, self.isMouseOver
        ## Add target cars
        if self.isClick:
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
                            self.bufferID.append([id, box])
                            id = self.chooseOneID()
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
                            removed_object = self.titles.pop(index_to_remove)
                            removed_carids = self.carids.pop(index_to_remove)

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
        if len(self.targetID) > 0:
            img0 = draw_boxes(
                img=img0,
                bbox=np.array(targetCars),
                identities=self.targetID,
                bg_im=self.bg_im,
                titles=self.titles,
                scale=1,
            )
        return img0, self.isMouseOver

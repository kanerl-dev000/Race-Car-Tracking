import cv2
from ultralytics import YOLO

import numpy as np

from utils.draw1 import draw_boxes


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

        self.ids = "hello"

        self.fontPath_driver = "./Font/AgencyFBBold.ttf"
        self.fontPath_num = "./Font/OstrichSans-Bold.otf"

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

    def checktrackid(self, id):
        for d in self.titles:
            if id == d["trackid"]:
                return True
            else:
                return False

    def findNoneid(self):
        for i, t in enumerate(self.titles):
            if t["trackid"] not in self.targetID:
                return i
        return len(self.titles)

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
        for title in self.titles:
            if (
                title["trackid"] in self.track_ids
                and title["trackid"] not in self.targetID
            ):
                self.targetID.append(title["trackid"])

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
                            self.targetID.append(id)

                            if self.checktrackid(id):
                                break
                            else:
                                if len(self.titles) > len(self.targetID):
                                    self.titles[len(self.targetID) - 1]["trackid"] = id
                                else:
                                    self.titles[self.findNoneid()]["trackid"] = id

                            for i, d in enumerate(self.titles):
                                if id == d["trackid"]:
                                    break
                                elif i == len(self.titles) - 1:
                                    self.titles[len(self.targetID) - 1]["trackid"] = id

                            break

                    else:
                        self.targetID.remove(id)
                        index = next(
                            (
                                i
                                for i, d in enumerate(self.titles)
                                if list(d.values())[0] == id
                            ),
                            None,
                        )
                        if index != None:
                            self.titles[index]["trackid"] = 0

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
                titles=self.titles,
                font_path_num=self.fontPath_num,
                font_path_drive=self.fontPath_driver,
            )
        return img0, self.isMouseOver

import cv2
from ultralytics import YOLO

import win32api
import win32con

import numpy as np

from draw import draw_boxes


class CarTrack(object):
    def __init__(self):
        # Load the YOLOv8 model
        self.model = YOLO("yolov8n.pt")
        # Open the video file
        self.video_path = "./videos/1.mp4"
        self.mouse = [-1000, -1000]

        self.isMouseOver = False
        self.isClick = False

        self.track_ids = []
        self.bufferID = []
        self.bufferBOX = []

        self.targetID = []

        self.title = {}

    def __enter__(self):
        pass
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def mousePoints(self, event, x, y, flags, param):
        # Left button mouse click event opencv
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse = [x, y]
            self.isClick = True
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse = [x, y]

    def xywh_to_xyxy(self, box):
        xx, yy, ww, hh = box
        return [xx - ww // 2, yy - hh // 2, xx + ww // 2, yy + hh // 2]

    def chooseOneID(self):
        minID = 0
        minDistance = 100000
        for bufferID in self.bufferID:
            x, y, w, h = bufferID[1]
            dist = (self.mouse[0] - x) ** 2 + (self.mouse[1] - y) ** 2
            print("dist", dist)
            if minDistance > dist:
                minDistance = dist
                minID = bufferID[0]
            print("minID", minID)
        return minID

    def run(self):
        cap = cv2.VideoCapture(self.video_path)

        bg_im = cv2.imread("./img/background.PNG")

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if self.isMouseOver:
                win32api.SetCursor(win32api.LoadCursor(0, win32con.IDC_SIZEALL))
            else:
                win32api.SetCursor(win32api.LoadCursor(0, win32con.IDC_CROSS))

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = self.model.track(frame, persist=True)
                boxes = results[0].boxes.xywh.cpu()

                if not results[0].boxes.id == None:
                    self.track_ids = results[0].boxes.id.int().cpu().tolist()
                else:
                    continue

                ## Add target cars
                if self.isClick:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = self.xywh_to_xyxy(box)
                        if self.mouse[0] in range(int(x1), int(x2)) and self.mouse[
                            1
                        ] in range(int(y1), int(y2)):
                            id = self.track_ids[i]
                            if id not in self.targetID:
                                self.bufferID.append([id, box])
                                id = self.chooseOneID()
                                self.targetID.append(id)
                            else:
                                self.targetID.remove(id)

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
                    if self.mouse[0] in range(int(x1), int(x2)) and self.mouse[
                        1
                    ] in range(int(y1), int(y2)):
                        self.isMouseOver = True
                        break
                    elif i == len(boxes) - 1:
                        self.isMouseOver = False

                ## Draw title
                targetCars = []
                for id in self.targetID:
                    targetCars.append(boxes[self.track_ids.index(id)])

                # if len(targetCars) > 0:

                # Visualize the results on the frame
                # annotated_frame = results[0].plot()
                # img = results[0].orig_img
                img = frame

                if len(self.targetID) > 0:
                    img = draw_boxes(
                        img=img,
                        bbox=np.array(targetCars),
                        identities=self.targetID,
                        bg_im=bg_im,
                        title=self.title,
                    )
                # Display the annotated frame
                cv2.imshow("test", img)
                cv2.setMouseCallback("test", self.mousePoints)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    with CarTrack() as car_track:
        car_track.run()

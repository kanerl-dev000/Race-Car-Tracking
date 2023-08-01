import os
import copy
import time

import cv2
import numpy as np

import onnxruntime

from utils import utils_onnx

from deep_sort import build_tracker


from utils.parser import get_config
from utils.draw import draw_boxes

import warnings

import torch


class CarTrack(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.deepsort = None
        self.target_cars = []
        self.mouse = [-1000, -1000]
        self.identities = []

        self.model_path = "weight/YOLOPv2.onnx"
        self.score_th = 0.7
        self.nms_th = 0.45
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

    def run_inference(self, onnx_session, image):
        input_image = copy.deepcopy(image)
        input_image, _, (pad_w, pad_h) = utils_onnx.letterbox(input_image)

        input_image = input_image[:, :, ::-1].transpose(2, 0, 1)

        input_image = np.ascontiguousarray(input_image)

        input_image = input_image.astype("float32")
        input_image /= 255.0

        input_image = np.expand_dims(input_image, axis=0)

        input_name = onnx_session.get_inputs()[0].name
        results = onnx_session.run(None, {input_name: input_image})

        result_dets = []
        result_dets.append(results[0][0])
        result_dets.append(results[0][1])
        result_dets.append(results[0][2])

        anchor_grid = []
        anchor_grid.append(results[1])
        anchor_grid.append(results[2])
        anchor_grid.append(results[3])

        result_dets = utils_onnx.split_for_trace_model(
            result_dets,
            anchor_grid,
        )

        result_dets = utils_onnx.non_max_suppression(
            result_dets,
            conf_thres=self.score_th,
            iou_thres=self.nms_th,
        )

        bboxes = []
        bbxywh = []
        scores = []
        class_ids = []
        for result_det in result_dets:
            if len(result_det) > 0:
                result_det[:, :4] = utils_onnx.scale_coords(
                    input_image.shape[2:],
                    result_det[:, :4],
                    image.shape,
                ).round()

                for *xyxy, score, class_id in reversed(result_det):
                    x1, y1 = xyxy[0], xyxy[1]
                    x2, y2 = xyxy[2], xyxy[3]
                    bbxywh.append(
                        np.array(
                            [int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)]
                        )
                    )
                    bboxes.append(np.array([int(x1), int(y1), int(x2), int(y2)]))
                    scores.append(float(score))
                    class_ids.append(int(class_id))

        return (np.array(bboxes), scores, class_ids, np.array(bbxywh))

    def main(self):
        self.deepsort = build_tracker(self.cfg, use_cuda=use_cuda)

        video_path = "sample.mp4"

        bg_im = cv2.imread("./img/background.PNG")

        if not os.path.isfile(self.model_path):
            import urllib.request

            url = "https://github.com/Kazuhito00/YOLOPv2-ONNX-Sample/releases/download/v0.0.0/YOLOPv2.onnx"
            save_path = "weight/YOLOPv2.onnx"

            print("Start Download:YOLOPv2.onnx")
            urllib.request.urlretrieve(url, save_path)
            print("Finish Download")

        onnx_session = onnxruntime.InferenceSession(
            self.model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        video_capture = cv2.VideoCapture(video_path)

        while True:
            start_time = time.time()

            ret, frame = video_capture.read()
            if not ret:
                break
            (bboxes, scores, class_ids, bbxywh) = self.run_inference(
                onnx_session, frame
            )

            elapsed_time = time.time() - start_time

            debug_image = self.draw_debug_image(
                frame, (bboxes, scores, class_ids, bbxywh), elapsed_time, bg_im
            )

            cv2.imshow("test", debug_image)
            cv2.setMouseCallback("test", self.mousePoints)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def draw_debug_image(self, image, car_dets, elapsed_time, bg_im):
        (bboxes, scores, class_ids, bbxywh) = car_dets
        outputs = []
        if len(bbxywh > 0):
            outputs = self.deepsort.update(bbxywh, scores, image)
        if len(outputs) > 0:
            # Filter selected cars
            self.target_cars = []
            for box in outputs:
                [x1, y1, x2, y2, id] = box
                w = x2 - x1
                h = y2 - y1

                if (
                    x1 + w // 2 < self.mouse[0]
                    and self.mouse[0] < x2 + w // 2
                    and y1 + h // 2 < self.mouse[1]
                    and self.mouse[1] < y2 + h // 2
                ):
                    if id not in self.identities:
                        self.identities.append(id)
                        break
                    else:
                        continue
            i = 0
            while i < len(self.identities):
                box_id = self.identities[i]
                if box_id not in outputs[:, 4]:
                    self.identities.remove(box_id)
                else:
                    index = (list(outputs[:, 4])).index(box_id)
                    self.target_cars.append(np.array(outputs[index][0:4]))
                    i += 1
            self.mouse = [-1000, -1000]
            if len(self.target_cars) > 0:
                image = draw_boxes(
                    img=image,
                    bbox=np.array(self.target_cars),
                    identities=self.identities,
                    bg_im=bg_im,
                    title=self.title,
                )

        return image


if __name__ == "__main__":
    cfg = get_config()
    cfg.merge_from_file("./configs/deep_sort.yaml")
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
    with CarTrack(cfg) as car_track:
        car_track.main()

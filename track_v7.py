
import os
import sys
import argparse
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from random import random as ran
from datetime import datetime

import pandas as pd
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

from draw import draw_boxes


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # strongsort root directory
WEIGHTS = ROOT / 'weights'


if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov7 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, set_logging
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier

from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT


class CarTrack(object):
    def __init__(self):
        self.source = './videos/'
        self.weights = './weights/yolov7-tiny.pt'
        self.show_vid = True
        self.image_size = 640
        self.classes = [2, 7]
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = ''
        self.agnostic_nms = ''
        self.augment = ""
        self.line_thickness = 1
        self.title = {}
        self.target_cars = []
        self.mouse = [-1000, -1000]
        self.identities = []



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

    def detect(self):
        bg_im = cv2.imread("./img/background.PNG")
        source = self.source
        weights = self.weights
        show_vid = self.show_vid
        image_size = self.image_size
        classes =  self.classes
        conf_thres = self.conf_thres
        iou_thres = self.iou_thres
        device = self.device
        agnostic_nms = self.agnostic_nms
        augment = self.augment

        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        strong_sort_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt'  # model.pt path,
        hide_class=True  # hide IDs
        line_thickness=self.line_thickness

        # Initialize
        set_logging()
        device = select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(image_size, s=stride)  # check img_size

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        if webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
            nr_sources = len(dataset)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
            nr_sources = 1

        # initialize StrongSORT
        cfg = get_config()
        cfg.merge_from_file("./strong_sort/configs/strong_sort.yaml")

        # Create as many strong sort instances as there are video sources
        strongsort_list = []
        for i in range(nr_sources):
            strongsort_list.append(
                StrongSORT(
                    strong_sort_weights,
                    device,
                    max_dist=cfg.STRONGSORT.MAX_DIST,
                    max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.STRONGSORT.MAX_AGE,
                    n_init=cfg.STRONGSORT.N_INIT,
                    nn_budget=cfg.STRONGSORT.NN_BUDGET,
                    mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                    ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
                )
            )
        outputs = [None] * nr_sources

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()

        # Run tracking
        curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

        for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                curr_frames[i] = im0
                p = Path(p)  # to Path

                s += '%gx%g ' % img.shape[2:]  # print string
                if cfg.STRONGSORT.ECC:  # camera motion compensation
                    strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to strongsort
                    outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                    # draw boxes for visualization
                    # if len(outputs[i]) > 0:
                    #     for j, (output, conf) in enumerate(zip(outputs[i], confs)):        
                    #         bboxes = output[0:4]
                    #         id = output[4]
                    #         cls = output[5]
                    #         c = int(cls)  # integer class
                    #         id = int(id)  # integer id
                    #         label = f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'
                    #         plot_one_box(bboxes, im0, label=label, color=colors[int(cls)], line_thickness=line_thickness)
                    # print(outputs)
                    if len(outputs) > 0:
                        # print(outputs[0])
                        # Filter selected cars
                        self.target_cars = []
                        for box in outputs[0]:
                            [x1, y1, x2, y2, id, class_id, score] = box
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
                else:
                    strongsort_list[i].increment_ages()
                    print('No detections')
    
                # Stream results
                if show_vid:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        break

                prev_frames[i] = curr_frames[i]

if __name__ == '__main__': 
    with torch.no_grad():
        with CarTrack() as car_track:
            car_track.detect()




import os
import cv2
import time
import argparse
import torch
import numpy as np

from detector import build_detector_advanced
from deep_sort import build_tracker
from utils.draw import draw_boxes,xyxy_to_xywh
from utils.parser import get_config

from threading import Thread
from imutils.video import WebcamVideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class VideoTracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            raise UserWarning("Running in cpu mode!")

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.stream = cv2.VideoCapture()
        self.detector = build_detector_advanced()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.stream.open(self.args.VIDEO_PATH)
        self.im_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        assert self.stream.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        idx_frame = 0
        print("[INFO] starting video file thread...")
        fvs = FileVideoStream(self.args.VIDEO_PATH).start()
        time.sleep(1.0)
        fps = FPS().start()
        while fvs.more():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.stream.retrieve()
            frame = fvs.read()
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xyxy, cls_conf, cls_ids = self.detector(im)
            if len(bbox_xyxy) != 0:
                bbox_xywh = xyxy_to_xywh(bbox_xyxy)
                if bbox_xywh is not None:
                    # select person class
                    mask = cls_ids == 0

                    bbox_xywh = bbox_xywh[mask]
                    bbox_xywh[:, 3:] *= 1.2  # bbox dilation just in case bbox too small
                    cls_conf = cls_conf[mask]

                    # do tracking
                    outputs = self.deepsort.update(bbox_xywh, cls_conf, im, True)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
            fps.update()
            if self.args.save_path:
                self.writer.write(ori_im)
            end = time.time()
            print("time: {:.03f}s, fps: {:.03f}".format(end - start, 1 / (end - start)))

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)


        fps.stop()
        fvs.stop()
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args) as vdo_trk:
        vdo_trk.run()
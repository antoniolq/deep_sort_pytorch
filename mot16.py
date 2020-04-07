import os
import cv2
import time
import argparse
import torch
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config


class imageTracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            raise UserWarning("Running in cpu mode!")

        self.dir = "/mnt/Disk1/qingl/data/MOT16/train/MOT16-02/img1/"
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def run(self):
        images = os.listdir(self.dir)
        idx_frame = 0
        while idx_frame >= len(images):
            idx_frame += 1
            start = time.time()
            _, ori_im = images[idx_frame]
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)
            if bbox_xywh is not None:
                # select person class
                mask = cls_ids==0

                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:,3:] *= 1.2 # bbox dilation just in case bbox too small
                cls_conf = cls_conf[mask]

                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
                f = open("/home/qingl/antonio/mot16/test.txt", 'a')
                for row in outputs:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (idx_frame,
                        row[0], row[1], row[2], row[3], row[4]), file=f)

            end = time.time()
            print("time: {:.03f}s, fps: {:.03f}".format(end-start, 1/(end-start)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with imageTracker(cfg, args) as img_trk:
        img_trk.run()

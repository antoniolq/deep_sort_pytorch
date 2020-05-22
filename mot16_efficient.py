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
from skimage import io

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class imageTracker(object):
    def __init__(self, cfg, args, name):
        self.cfg = cfg
        self.args = args
        self.name = name
        self.dir = self.args.dir
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            raise UserWarning("Running in cpu mode!")

        self.indir = "/mnt/Disk1/qingl/data/MOT16/train/"+ self.name +"/img1/"
        self.outdir = "/home/qingl/antonio/deep_sort_pytorch/mot16/advanced/" + self.dir + self.name +".txt"
        f = open(self.outdir, 'w')
        f.truncate()
        self.detector = build_detector_advanced()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)


    def run(self):
        images = os.listdir(self.indir)
        idx_frame = 0
        imgs = sorted(images)
        print(imgs[0])
        while idx_frame < len(imgs):
            tmp = self.indir + imgs[idx_frame]
            img = io.imread(tmp)
            start = time.time()
            im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                    results = self.deepsort.update(bbox_xywh, cls_conf, im, False)
                    f = open(self.outdir, 'a')
                    for row in results:
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (idx_frame,
                            row[0], row[1], row[2], row[3], row[4]), file=f)
            print(imgs[idx_frame],"finished")
            idx_frame += 1
            end = time.time()
            print("time: {:.03f}s, fps: {:.03f}".format(end-start, 1/(end-start)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--name", type=str, default="MOT16-02")
    parser.add_argument("--dir", type=str, default="eff0")
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    list = ["MOT16-02","MOT16-04","MOT16-05","MOT16-09","MOT16-10","MOT16-11","MOT16-13"]
    id = 0
    while(id < len(list)):
        print(list[id], ".txt started------------")
        with imageTracker(cfg, args, list[id]) as img_trk:
            img_trk.run()
        print(list[id], ".txt finished-----------")
        id += 1
    print("all finished")
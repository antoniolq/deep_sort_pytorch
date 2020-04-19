from .YOLOv3 import YOLOv3
from .EfficientDet import EfficientDet


__all__ = ['build_detector', 'build_detector_advanced']

def build_detector(cfg, use_cuda):
    return YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES,
                    score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH,
                    is_xywh=True, use_cuda=use_cuda)


def build_detector_advanced():
    return EfficientDet()
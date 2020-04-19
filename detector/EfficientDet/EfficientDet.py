import time

import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from tools.utils import preprocess, invert_affine, postprocess

class EfficientDet(object):
    def __init__(self):
        # net definition
        self.compound_coef = 7
        self.force_input_size = None  # set None to use default size
        self.img_path = '"/home/qingl/antonio/deep_sort_pytorch/detector/EfficientDet/imgs/000128.jpg"'

        # replace this part with your project's anchor config
        self.anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        self.anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

        self.threshold = 0.2
        self.iou_threshold = 0.2

        self.use_cuda = True
        self.use_float16 = False
        cudnn.fastest = True
        cudnn.benchmark = True
        self.obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light',
                    'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                    'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                    'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife',
                    'spoon',
                    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                    'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                    'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                    'toothbrush']

        # tf bilinear interpolation is different from any other's, just make do
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.input_size = self.input_sizes[self.compound_coef] if self.force_input_size is None else self.force_input_size

    def __call__(self):
        print("callll")

        ori_imgs, framed_imgs, framed_metas = preprocess(self.img_path, max_size=self.input_size)

        if self.use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not self.use_float16 else torch.float16).permute(0, 3, 1, 2)

        model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=len(self.obj_list),
                                     ratios=self.anchor_ratios, scales=self.anchor_scales)
        model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
        model.requires_grad_(False)
        model.eval()

        if self.use_cuda:
            model = model.cuda()
        if self.use_float16:
            model = model.half()

        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              self.threshold, self.iou_threshold)
        out = invert_affine(framed_metas, out)
        self.display(out, ori_imgs, imshow=False, imwrite=True)

        print('running speed test...')
        with torch.no_grad():
            print('test1: model inferring and postprocessing')
            print('inferring image for 10 times...')
            t1 = time.time()
            for _ in range(10):
                _, regression, classification, anchors = model(x)

                out = postprocess(x,
                                  anchors, regression, classification,
                                  regressBoxes, clipBoxes,
                                  self.threshold, self.iou_threshold)
                out = invert_affine(framed_metas, out)

            t2 = time.time()
            tact_time = (t2 - t1) / 10
            print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

    def display(self, preds, imgs, imshow=True, imwrite=True):
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                continue

            for j in range(len(preds[i]['rois'])):
                (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
                cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
                obj = self.obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])

                cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                            (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 1)

            if imshow:
                cv2.imshow('img', imgs[i])
                cv2.waitKey(0)

            if imwrite:
                cv2.imwrite(f'imgs/result.jpg', imgs[i])
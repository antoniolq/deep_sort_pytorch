import torch
import numpy as np


def Diou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:  #
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True
    # #xmin,ymin,xmax,ymax->[:,0],[:,1],[:,2],[:,3]
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    return dious

def python_nms(boxes, scores, nms_thresh):
    """ Performs non-maximum suppression using numpy
        Args:
            boxes(Tensor): `xyxy` mode boxes, use absolute coordinates(not support relative coordinates),
                shape is (n, 4)
            scores(Tensor): scores, shape is (n, )
            nms_thresh(float): thresh
        Returns:
            indices kept.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    # Use numpy to run nms. Running nms in PyTorch code on CPU is really slow.
    origin_device = boxes.device
    cpu_device = torch.device('cpu')
    boxes = boxes.to(cpu_device).numpy()
    scores = scores.to(cpu_device).numpy()

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]
    num_detections = boxes.shape[0]
    for _i in range(num_detections):
        i = order[_i]
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]

        for _j in range(_i + 1, num_detections):
            j = order[_j]
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)

            x_center = xx1 + (xx2 - xx1) / 2
            y_center = yy1 + (yy2 - yy1) / 2
            x1_center = x1[j] + (x2[j] - x1[j]) / 2
            y1_center = y1[j] + (y2[j] - y1[j]) / 2
            inter_diag = (x1_center - x_center) ** 2 + (y1_center - y_center) ** 2
            outer_diag = (x2[j] - ix1) ** 2 + (y2[j] - iy1) ** 2
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            dious = ovr - inter_diag / outer_diag
            if dious >= nms_thresh:
                scores[j] = np.exp(-(ovr * ovr) / 0.5) * scores[j]
    keep = np.where(scores > nms_thresh)[0]
    keep = torch.from_numpy(keep).to(origin_device)
    # print("yes")
    return keep

import sys 
sys.path.append('.')

import os
import argparse

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn

# from data_preparation.yolov5_data_copy import LoadImages_try
from data_preparation.yolov5_data import LoadImages, LoadImages1
from detect_.yolov5_test_utils import red_detector, create_feature, Control, standardize, output_tr_image

from networks.yolov5 import YOLOv5Model

from collections import OrderedDict
import cv2
import numpy as np
import time
import random
from matplotlib.patches import Ellipse
from shapely.geometry import Polygon

import json



def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output

# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE:
#         # 在控制台打印鼠标位置的坐标
#         print("Mouse Position: x={}, y={}".format(x, y))

# def yolo_detect_test(image,preds,hr,wr,w0):
#     brake = None
#     throttle = None

#     veh_detect = None
#     tr_detect = None

#     # # width limit of traffic lights
#     w_min_tr = 460 # int(w0*0.45)
#     w_max_tr = 543  #int(w0*0.535) 

#     # # height limit of traffic lights
#     h_min_tr = 168  # int(h0*0.3)
#     h_max_tr = 186  # int(h0*0.4)

#     cv2.namedWindow('1')

#     polygon_points = np.array([[260,480], [700,480], [590,380],[510,340],[480, 300],[450,340] ,[370,380]], np.int32)
#     h_min_polygon = min(polygon_points[:, 1])
#     polygon = Polygon(polygon_points)
#     exterior_coords0 = list(polygon.exterior.coords)
#     contour0 = np.array(exterior_coords0, dtype=np.int32)
#     cv2.drawContours(image, [contour0], 0, (0, 0, 250), thickness=3)

#     for pred in preds:
#         print(pred)
#         # define parameters
#         w_min = int(pred[0]*wr)
#         w_max = int(pred[2]*wr)
#         h_min = int(pred[1]*hr)
#         h_max = int(pred[3]*hr)
        
#         # for detecting traffic light 
#         if pred[5] == 1:
#             # print(w_min,w_max,h_min, h_max)
#             if w_min_tr< (w_min + w_max)/2 < w_max_tr and h_min_tr < (h_min + h_max)/2 < h_max_tr:
#                 img_tr = standardize(image[(h_min-3):(h_max+3), (w_min-3):(w_max+3)])
#                 label = red_detector(img_tr)
#                 if label[0] == 1:
#                     tr_detect = True
                    
#         # for detecting bus and car
#         if pred[5] == 2 or pred[5] == 0:
#             if 470 > h_max > h_min_polygon:
#                 center = (int((w_min+ w_max)/2), int((h_min+ h_max)/2))
#                 width = abs(w_max - w_min)
#                 height = abs(h_min- h_max)

#                 ellipse = Ellipse(center, width, height, angle=0) 
#                 vertices = ellipse.get_verts()     # get the vertices from the ellipse object
#                 ellipse = Polygon(vertices)
#                 exterior_coords2 = list(ellipse.exterior.coords)
#                 contour2 = np.array(exterior_coords2, dtype=np.int32)
#                 cv2.drawContours(image, [contour2], 0, (0, 255, 0), thickness=2)

#                 if polygon.intersects(ellipse):
#                     veh_detect = True

#         cv2.rectangle(image,(w_min_tr, h_min_tr), (w_max_tr, h_max_tr), color=(0,0,255), thickness=1)
#         cv2.rectangle(image,(564,184), (621,196), color=(0,0,255), thickness=1)
#         cv2.rectangle(image,(282,121), (309,161), color=(0,0,255), thickness=1)
#         if not pred[5] == 3:
#             cv2.rectangle(image,(int(pred[0]*wr), int(pred[1]*hr)), (int(pred[2]*wr), int(pred[3]*hr)), color=(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)), thickness=2)
    
#     cv2.setMouseCallback('1', mouse_callback)
#     cv2.imshow("1",image)
#     if tr_detect == True:
#         brake = 1
#         throttle = 0
#         print("The traffic light is red, so we set our brake to 1 and our throttle to 0.")
#     else:
#         if veh_detect == True:
#                 brake = 1
#                 throttle = 0
#                 print("A vehicle is in front of us, so we set our brake to 1 and our throttle to 0.")
#         else:
#             print("No vehicle in range. We can pass.")
#             # pass
#     cv2.waitKey()
#     return brake, throttle


def yolo_detect(image,preds,hr,wr,w0):
    brake = None
    throttle = None

    # im, im0s = dataset
    veh_detect = None
    tr_detect = None

    # # image width limit of traffic lights
    w_min_tr = int(w0*0.45) #576 # 
    w_max_tr = int(w0*0.55) #684  #

    # # # the limit of traffic lights detection, if below this limit, we do red deteciton, if not, the distance is long, do not need to brake
    h_min_tr = int(480*0.2)
    h_max_tr = int(480*0.42)

    # polygon_points = np.array([[260,480], [700,480], [590,380],[510,340],[480, 300],[450,340] ,[370,380]], np.int32) # s10
    polygon_points = np.array([[410,480], [870,480], [750,380], [670,340], [640,260] ,[610,340], [530,380]], np.int32) # s30
    h_min_polygon = min(polygon_points[:, 1])
    polygon = Polygon(polygon_points)
    exterior_coords0 = list(polygon.exterior.coords)
    contour0 = np.array(exterior_coords0, dtype=np.int32)
    cv2.drawContours(image, [contour0], 0, (255, 255, 255), thickness=1)

    for pred in preds:
        # define parameters
        w_min = int(pred[0]*wr)
        w_max = int(pred[2]*wr)
        h_min = int(pred[1]*hr)
        h_max = int(pred[3]*hr)

        # for detecting traffic light 
        if pred[5] == 1:
            if w_min_tr< (w_min + w_max)/2 < w_max_tr and h_min_tr < (h_min + h_max)/2 < h_max_tr:
                img_tr = standardize(image[(h_min-3):(h_max+3), (w_min-3):(w_max+3)])
                label = red_detector(img_tr)
                if label[0] == 1:
                    tr_detect = True
                    
        # for detecting bus and car
        if pred[5] == 2 or pred[5] == 0:
            if 470 > h_max > h_min_polygon:
                center = (int((w_min+ w_max)/2), int((h_min+ h_max)/2))
                width = abs(w_max - w_min)
                height = abs(h_min- h_max)

                ellipse = Ellipse(center, width, height, angle=0) 
                vertices = ellipse.get_verts()     # get the vertices from the ellipse object
                ellipse = Polygon(vertices)
                exterior_coords2 = list(ellipse.exterior.coords)
                contour2 = np.array(exterior_coords2, dtype=np.int32)
                cv2.drawContours(image, [contour2], 0, (0, 255, 0), thickness=1)

                if polygon.intersects(ellipse):
                    veh_detect = True

        cv2.rectangle(image,(w_min_tr, h_min_tr), (w_max_tr, h_max_tr), color=(255,255,255), thickness=1)
        if not pred[5] == 3:
            # cv2.rectangle(image,(int(pred[0]*wr), int(pred[1]*hr)), (int(pred[2]*wr), int(pred[3]*hr)), color=(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)), thickness=2)
            pass
    if tr_detect == True:
        brake = 1
        throttle = 0
        print("The traffic light is red: brake=1, throttle=0!!!!!!!!!!!!!!!!!!")
    else:
        if veh_detect == True:
                brake = 1
                throttle = 0
                print("A vehicle is in front of us: brake=1, throttle=0!!!!!!!!!!!!!!!!!!!!")
        else:
            # throttle = 0.3
            # brake = 0.5
            print("No interference.")
            # pass
    return brake, throttle



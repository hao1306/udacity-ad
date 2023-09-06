# 这个控制的总体思路就是接收每一帧模拟器的图片数据，然后将图片输入我们的神经网络模型得到一个输出（指令），再将这些指令打包发回给模拟器实现对模拟器车辆的控制。

# 代码里首先接收一个叫"send_image"的事件（41行），这个事件是模拟器发送过来的，里面包含了图片信息。
# 因为Socket网络通信必须发json类型的数据，因此发送过来的图片信息需要一系列解码操作才能使用（46-49行）。

# 接下来就是将图片编程适合我们模型输入的格式，然后输入到我们的模型得到输出，并执行函数send_control，将指令打包成json格式发回给模拟器（63-71行）

import socketio
import socket
# concurrent networking
import eventlet

# web server gateway interface
import eventlet.wsgi
from flask import Flask
import base64
import cv2
import numpy as np

# from io import BytesIO
import time
import os
from io import BytesIO

# image processing and model
from PIL import Image
import torch
from torch import nn
from networks.resnet import resnet_model
from networks.yolov5 import YOLOv5Model
import utils
from collections import OrderedDict
import random
from matplotlib.patches import Ellipse
from shapely.geometry import Polygon
from data_preparation.yolov5_data import LoadImages1
from detect_.yolov5_test_final import non_max_suppression, yolo_detect

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "1"

device0 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device1 = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# 定义模型
model = resnet_model()
# 导入权重
model = torch.load('weights/resnet/349.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device1)
model.eval()
# import yolov5 model
model_yolo = YOLOv5Model()

for m in model_yolo.modules():
    t = type(m)
    if t is nn.Conv2d:
        pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif t is nn.BatchNorm2d:
        m.eps = 1e-3
        m.momentum = 0.03
    elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
        m.inplace = True

ckpt = torch.load('weights/yolov5/best_02.pt', map_location='cpu')

weights = OrderedDict()

for name1, name2 in zip(ckpt, model_yolo.state_dict()):
    weights[name2] = ckpt[name1]

model_yolo.load_state_dict(weights)
# model_yolo = model_yolo.half()
model_yolo = model_yolo.to(device1)
model_yolo.eval()

# brake = None
# throttle = None
# steering_angle = None

path = 'data/simulator/Dataset100/IMG/CapturedImage1776.jpg'
# print(os.path.isfile(path))
img = cv2.imread(path)
img_resnet = img.astype(np.float32)/255
# Calculate and print fps
# image = Image.open(BytesIO(base64.b64decode(img_data)))
# img = np.asarray(image)
img_resnet = utils.preprocess_drive(img_resnet)
img_resnet = torch.from_numpy(img_resnet).permute(2,0,1)
img_resnet = img_resnet.unsqueeze(0)

# resnet part
img_resnet = img_resnet.to(device1)
command = model(img_resnet)
throttle = command[0, 0].item()
brake = command[0, 1].item()
steering_angle = command[0, 2].item()

# yolov5 part
im, im0s = LoadImages1(img)
h,w = im.shape[1:]
h0,w0 = im0s.shape[:2]
hr = h0/h
wr = w0/w

im = torch.from_numpy(im)#.to(model.device)
# im = im.half() if model_yolo.fp16 else im.float()  # uint8 to fp16/32
im = im.float()
im /= 255  # 0 - 255 to 0.0 - 1.0
if len(im.shape) == 3:
    im = im[None]  # expand for batch dim

im = im.to(device1)
preds = model_yolo(im)
preds = non_max_suppression(preds, conf_thres = 0.25, iou_thres = 0.45)
preds0 = preds[0].detach()
preds0 = preds0.cpu()
preds0 = preds0.numpy()
brake_yolo, throttle_yolo = yolo_detect(im0s,preds0,hr,wr,w0)
print(type(brake), brake, 'brake')
print(type(brake_yolo), brake_yolo, 'brake_yolo')
if brake_yolo is not None:
    brake = brake_yolo
if throttle_yolo is not None:
    throttle = throttle_yolo

print('steering_angle: {} throttle: {} brake: {}'.format(steering_angle, throttle, brake))


    #这一部分是算帧率的部分，可根据需求选择保不保留
    # frame_count += 1
    # elapsed_time = time.time() - prev_time
    # if elapsed_time > 1:
    #     fps = frame_count / elapsed_time
    #     print(f"FPS: {fps:.2f}")
    #     prev_time = time.time()
    #     frame_count = 0

    # show the recieved images on the screen

    # if img is not None and img.shape[0] > 0 and img.shape[1] > 0:

    #     image_info = np.array(img).astype(np.float32)/255
    #     image_info = torch.from_numpy(image_info).permute(2,0,1)
    #     result = model(image_info.unsqueeze(0))
    #     throttle = float(result[0][0])
    #     brake = float(result[0][1])
    #     steering_angle = float(result[0][2])
    #     send_control(steering_angle, throttle, brake)

    #     #这一部分是显示从模拟器接收到的每一帧图像，可根据需求选择保不保留
    #     cv2.namedWindow("image from unity", cv2.WINDOW_NORMAL)
    #     cv2.imshow("image from unity", img)
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         cv2.destroyAllWindows()
    #         return
    # else:
    #     print("Invalid image data")



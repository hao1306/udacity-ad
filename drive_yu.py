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
from detect_.yolov5_test_final_mo import non_max_suppression, yolo_detect

# Initialize Socket.IO server
sio = socketio.Server()
app = Flask(__name__)

#这一部分定义是为了算这个drive服务端脚本接收图片的帧率可根据需求选择保不保留
frame_count = 0
frame_count_save = 0
prev_time = 0
fps = 0
model = None
model_yolo = None
counter = 0

@sio.on("send_image")
def on_image(sid, data):
    if data:
        # make the variables global to calculate the fps
        global frame_count, frame_count_save, prev_time, fps, counter
        counter += 1

        # print("image recieved!")
        img_data = data["image"]
        img_bytes = base64.b64decode(img_data)
    # Decode image from base64 format，将字典里抽取出来的字符串转换为字节串类型
        img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        img_resnet = img.astype(np.float32)/255
    # Calculate and print fps
        # image = Image.open(BytesIO(base64.b64decode(img_data)))
        # img = np.asarray(image)
        img_resnet = utils.preprocess_drive(img_resnet)
        img_resnet = torch.from_numpy(img_resnet).permute(2,0,1)
        img_resnet = img_resnet.unsqueeze(0)
        with torch.no_grad():
            try:
                # resnet part
                img_resnet = img_resnet.to(device)
                
                # yolov5 part
                im, im0s = LoadImages1(img)
                h,w = im.shape[1:]
                h0,w0 = im0s.shape[:2]
                hr = h0/h
                wr = w0/w
                
                im = torch.from_numpy(im)#.to(model.device)
                # im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im = im.float()
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

                im = im.to(device)
                preds = model_yolo(im)
                preds = non_max_suppression(preds, conf_thres = 0.25, iou_thres = 0.45)
                preds0 = preds[0].detach()
                preds0 = preds0.cpu()
                preds0 = preds0.numpy()

                if counter < 780:
                    steering_angle = 0
                    throttle = 1
                    brake = 0

                    # yolo control
                    brake_yolo, throttle_yolo = yolo_detect(im0s,preds0,hr,wr,w0)
                    if brake_yolo is not None:
                        brake = brake_yolo
                        steering_angle = 0
                    if throttle_yolo is not None:
                        throttle = throttle_yolo
                    # print('steering_angle: {} throttle: {} brake: {}'.format(steering_angle, throttle, brake))
                    print('steering_angle: {} throttle: {} brake: {} counter: {}'.format(steering_angle, throttle, brake, counter))
                    send_control(steering_angle, throttle, brake)
                else:
                    # resnet control
                    command = model(img_resnet)
                    throttle = command[0, 0].item()
                    brake = command[0, 1].item()
                    steering_angle = command[0, 2].item()
                    # yolov5 control
                    brake_yolo, throttle_yolo = yolo_detect(im0s,preds0,hr,wr,w0)
                    if brake_yolo is not None:
                        brake = brake_yolo
                    if throttle_yolo is not None:
                        throttle = throttle_yolo
                    print('steering_angle: {} throttle: {} brake: {} counter: {}'.format(steering_angle, throttle, brake, counter))
                    send_control(steering_angle, throttle, brake)
                torch.cuda.empty_cache()
            except Exception as e:
                print(e)

    else:
        sio.emit('manual', data={}, skip_sid=True)

    #这一部分是算帧率的部分，可根据需求选择保不保留
    frame_count += 1
    elapsed_time = time.time() - prev_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        prev_time = time.time()
        frame_count = 0

    # show the recieved images on the screen
    
    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
        # processing_path = "data\\simulator\\train\\Dataset_\\6\\" + str(counter) + ".jpg"
        # print(processing_path)
        # cv2.imwrite(processing_path, img)

        #这一部分是显示从模拟器接收到的每一帧图像，可根据需求选择保不保留
        cv2.putText(img, 'steering_angle: {} throttle: {} brake: {} counter: {}'.format(steering_angle, throttle, brake, counter), (440, 50), cv2.FONT_HERSHEY_SIMPLEX, .5 ,color=(255,255,255), thickness=1)
        cv2.namedWindow("image from unity", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("image from unity", img)
        key = cv2.waitKey(1) 
        if key == ord('q') or counter % 10 == 0:
            cv2.destroyAllWindows()
            return
        pass
    else:
        print("Invalid image data")


# 接收一个名为vehicle_data的事件，用于查看汽车的实时行驶数据
@sio.on("vehicle_data")
def vehicle_command(sid, data):
    print("data recieved!")
    steering_angle = float(data["steering_angle"])
    throttle = float(data["throttle"])
    brake = float(data["brake"])
    velocity = float(data["velocity"])

    print(f"parameters of the car: {velocity, steering_angle, throttle, brake}")
    # get image from the veichle

    if not data:
        print("data is empty")
    elif velocity < 1:
        send_control(0, 1, 0)


@sio.event
def connect(sid, environ):
    # sid for identifying the client connected表示客户端唯一标识符，environ表示其连接的相关环境信息
    print("Client connected")
    send_control(0, 0, 0)


# 这个函数作用是将预测指令打包发回给模拟器以便控制车辆驾驶
def send_control(steering_angle, throttle, brake):
    sio.emit(
        "control_command",
        data={
            "steering_angle": steering_angle.__str__(),
            "throttle": throttle.__str__(),
            "brake": brake.__str__(),
        },
        skip_sid=True,
    )


@sio.event
def disconnect(sid):
    # implement this function, if disconnected
    print("Client disconnected")


app = socketio.Middleware(sio, app)
# Connect to Socket.IO client
if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 定义模型
    model = resnet_model()
    # 导入权重
    model = torch.load('weights/resnet/349.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
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
    model_yolo = model_yolo.to(device)
    model_yolo.eval()

    # get local ip address
    ip = socket.gethostbyname(socket.gethostname())
    # 创建服务器实例
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)

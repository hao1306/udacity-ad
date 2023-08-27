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
from torch.autograd import Variable
from networks.resnet import resnet_model
import utils

# Initialize Socket.IO server
sio = socketio.Server()
app = Flask(__name__)

#这一部分定义是为了算这个drive服务端脚本接收图片的帧率可根据需求选择保不保留
frame_count = 0
frame_count_save = 0
prev_time = 0
fps = 0
model = None


@sio.on("send_image")
def on_image(sid, data):
    if data:
        # make the variables global to calculate the fps
        global frame_count, frame_count_save, prev_time, fps
        # print("image recieved!")
        img_data = data["image"]
    # img_bytes = base64.b64decode(img_data)
    # # Decode image from base64 format，将字典里抽取出来的字符串转换为字节串类型
    # img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    # Calculate and print fps
        image = Image.open(BytesIO(base64.b64decode(img_data)))
        img = np.asarray(image)
        img = utils.preprocess(img)
        try:
            img = Variable(torch.cuda.FloatTensor([img])).permute(0, 3, 1, 2)
            command = model(img)
            throttle = command[0].item()
            brake = command[1].item()
            steering_angle = command[2].item()
            print('steering_angle: {} throttle: {} brake: {}'.format(steering_angle, throttle, brake))
            send_control(steering_angle, throttle, brake)
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

        image_info = np.array(img).astype(np.float32)/255
        image_info = torch.from_numpy(image_info).permute(2,0,1)
        result = model(image_info.unsqueeze(0))
        throttle = float(result[0][0])
        brake = float(result[0][1])
        steering_angle = float(result[0][2])
        send_control(steering_angle, throttle, brake)

        #这一部分是显示从模拟器接收到的每一帧图像，可根据需求选择保不保留
        cv2.namedWindow("image from unity", cv2.WINDOW_NORMAL)
        cv2.imshow("image from unity", img)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            return
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
    elif velocity < 5:
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

    # get local ip address
    ip = socket.gethostbyname(socket.gethostname())
    # 创建服务器实例
    eventlet.wsgi.server(eventlet.listen((ip, 4567)), app)

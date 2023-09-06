import os
import cv2
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn


from matplotlib.patches import Ellipse
from shapely.geometry import Polygon

###### image pre-processing ######
# do image standardization, size = (32, 32)
def standardize(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st_img = cv2.resize(image, (32, 32))
    return st_img

###### traffic lights output ######
def output_tr_image(dictionary, folder_path):
    
    if dictionary is None:
        print("Error: Unable to output the image.")
    else:
        ## create traffic light folder
        # Specify the path and folder name
        # folder_path = os.path.dirname(dataset_path)
        folder_name = 'traffic_lights'
        full_path = os.path.join(folder_path, folder_name)

         # Check if the folder already exists
        if not os.path.exists(full_path):
            # Create the folder
            os.mkdir(full_path)
            print(f"Folder '{full_path}' created.")
        else:
            print(f"Folder '{full_path}' already exists.")

        for key in list(dictionary.keys()):
            # Perform color inversion
            image = dictionary[key]

            # Save the processed image
            processed_image_path = full_path + "/" + os.path.basename(key) + ".jpg"
        
            # print(processed_image_path)
            cv2.imwrite(processed_image_path, image)

        print("Image processing complete.")


###### red detection ######
def hsv_avg(rgb_image):
    # Convert to HSV
        # hue (color type), 
        # saturation (intensity of the color)
        # value (brightness or lightness)
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    area =hsv.shape[0]*hsv.shape[1] #pixels

    # HSV channels
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    #H channel
    h_sum_brightness = np.sum(h)
    h_avg = h_sum_brightness/area
        
    #S channel
    s_sum_brightness = np.sum(s)
    s_avg = s_sum_brightness/area
        
    #V channel
    v_sum_brightness = np.sum(v)
    v_avg = v_sum_brightness/area

    return h_avg, s_avg, v_avg

def create_mask_image(rgb_image,label):
    
    #Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    #analyze histogram
    if label == 'red':
        red_mask1 = cv2.inRange(hsv, (0,30,50), (10,255,255))
        red_mask2 = cv2.inRange(hsv, (150,40,50), (180,255,255))
        mask = cv2.bitwise_or(red_mask1,red_mask2)
        
    elif label == 'yellow':
        mask = cv2.inRange(hsv, (10,10,110), (30,255,255))
    
    #green
    else:
        mask = cv2.inRange(hsv, (45,40,120), (95,255,255))
    
    res = cv2.bitwise_and(rgb_image,rgb_image,mask = mask)
    
    return res

def create_feature(rgb_image):

    h,s,v = hsv_avg(rgb_image)
    image = np.copy (rgb_image)
    
    #apply mask
    red_mask = create_mask_image(image,'red')
    yellow_mask = create_mask_image(image,'yellow')
    green_mask = create_mask_image(image,'green')
    
    #slice into 3 parts, up, middle, down
    up = red_mask[0:10, :, :]
    middle = yellow_mask[11:20, :, :]
    down = green_mask[21:32, :, :]
    
    #find out hsv values based on each of the 3 parts
    h_up, s_up, v_up = hsv_avg(up)
    h_middle, s_middle, v_middle = hsv_avg(middle)
    h_down, s_down, v_down = hsv_avg(down)
    
    #v in hsv can detect whether theres value in up,middle or down
    if  v_up> v_middle and v_up> v_down:# and s_up>s_middle and s_up>s_down:
        return [1,0,0] #red
   
    elif  v_middle > v_down:# and s_middle>s_down:
        return [0,1,0] #yellow
 
    return [0,0,1] #green


def just_slice(rgb_image):
    
    image = np.copy (rgb_image)
    
    #slice into 3 parts, up, middle, down
    up = image[0:10, :, :]
    middle = image[11:20, :, :]
    down = image[21:32, :, :]
    
    #find out hsv values based on each of the 3 parts
    
    h_up, s_up, v_up = hsv_avg(up)
    h_middle, s_middle, v_middle = hsv_avg(middle)
    h_down, s_down, v_down = hsv_avg(down)
    
    #v in hsv can detect whether theres value in up,middle or down
    if  v_up> v_middle and v_up> v_down:# and s_up>s_middle and s_up>s_down:
            
        return [1,0,0] #red
   
    elif  v_middle > v_down:# and s_middle>s_down:
        return [0,1,0] #yellow
 
    return [0,0,1] #green

def red_mask_feature(rgb_image):
    
    #Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    #red_mask1 = cv2.inRange(hsv, (0,70,50), (10,255,255))
    #red_mask2 = cv2.inRange(hsv, (170,70,50), (180,255,255))
    #red_mask = cv2.bitwise_or(red_mask1,red_mask2)
        
        
    lower_red = np.array([150,140,140])
    upper_red = np.array([180,255,255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_result = cv2.bitwise_and(rgb_image, rgb_image, mask = red_mask)
     
    #find out hsv values based on each of the 3 parts
    h,s,v = hsv_avg(red_result)
    return h, s, v



def findNoneZero(rgb_image):
    rows,cols,_ = rgb_image.shape
    counter = 0
    for row in range(rows):
        for col in range(cols):
            pixels = rgb_image[row,col]
            if sum(pixels)!=0:
                counter = counter+1
    return counter

def rgb_detector(rgb_image):
    
    # Covert to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    area =hsv.shape[0]*hsv.shape[1] #pixels

    #S channel
    s_sum_brightness = np.sum(hsv[:,:,1])
    s_avg = s_sum_brightness/area

    # Green
    lower_green = np.array([70,140,140])
    upper_green = np.array([100,255,255])
    green_mask = cv2.inRange(hsv,lower_green,upper_green)
    green_result = cv2.bitwise_and(rgb_image,rgb_image,mask = green_mask)

    # Yellow
    lower_yellow = np.array([10,140,140])
    upper_yellow = np.array([60,255,255])
    yellow_mask = cv2.inRange(hsv,lower_yellow,upper_yellow)
    yellow_result = cv2.bitwise_and(rgb_image,rgb_image,mask=yellow_mask)
    
    # Red 
    lower_red = np.array([150,140,140])
    upper_red = np.array([180,255,255])
    red_mask = cv2.inRange(hsv,lower_red,upper_red)
    red_result = cv2.bitwise_and(rgb_image,rgb_image,mask = red_mask)

    # calculation sum
    sum_green = findNoneZero(green_result)
    sum_red = findNoneZero(red_result)
    sum_yellow = findNoneZero(yellow_result)
    if sum_red >= sum_yellow and sum_red>=sum_green:
        return [1,0,0]#Red
    if sum_yellow>=sum_green:
        return [0,1,0]#yellow
    return [0,0,1]#green

def red_detector(rgb_image):
    label1 = create_feature(rgb_image)
    label3 = rgb_detector(rgb_image)
    h,s,v= red_mask_feature(rgb_image)

    if label1 != [1,0,0]  and h>0:
        if h>0:
            label = [1,0,0]
    elif label1 == label3 and h == 0.0:
        label = label1
    else:
        label = label1
        
    return label

        

 
class Control:
    def __init__(self, label):
        # the label output from red_detector
        self.label = label

    def get_color(self):
        if self[0] == 1:
            color = 'red'
        elif self[1] == 1:
            color = 'yellow'
        else:
            color = 'green'
        return color
    
    def get_brake(self):
        brake = 0
        if self[0] == 1:
            brake = 1
        elif self[1] == 1:
            brake = 0
        else:
            brake = 0
        return brake
    
    def get_throttle(self):
        throttle = 1
        if self[0] == 1:
            throttle = 0
        elif self[1] == 1:
            throttle = 1
        else:
            throttle = 1
        return throttle

    # def __str__(self):
    #     return f"{self.label}"

# def get_detection(dictionary):


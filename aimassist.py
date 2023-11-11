#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyautogui
import pydirectinput
import time
import numpy as np
import win32gui
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt


# In[2]:


detector = hub.load('detector')


# In[3]:


image_data = tf.io.read_file("tomb.png")
image_decoded = tf.image.decode_image(image_data, channels = 3)
image_resized = tf.image.resize(image_decoded, [640, 640])
image_tensor = tf.expand_dims(image_resized, 0)
image_tensor = tf.cast(image_tensor, tf.uint8)


# In[4]:


COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


# In[5]:


def shoot():
    for i in range(10):
        pydirectinput.mouseDown(button='left')
        time.sleep(0.05)
        pydirectinput.mouseUp(button='left')


# In[6]:


def detectObject():
    ss = pyautogui.screenshot()
    #ss = ss.resize((640, 640))
    image_tensor = image_tensor = tf.convert_to_tensor(ss, dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    detector_output = detector(image_tensor)


    # Extract detection scores and classes
    detection_scores = detector_output["detection_scores"].numpy()[0]
    detection_classes = detector_output["detection_classes"].numpy()[0]

    # Filtering detections based on a threshold
    threshold = 0.5
    detected_indices = detection_scores > threshold

    # Filtered class IDs based on threshold
    detected_class_ids = detection_classes[detected_indices]

    # Get and print the class names of the detected objects
    detected_class_names = [COCO_CLASSES[int(class_id)] for class_id in detected_class_ids]

    for class_name in detected_class_names:
        print(class_name)
        if class_name != "person":
            shoot()


# In[7]:


def get_active_window_title():
    hwnd = win32gui.GetForegroundWindow()
    return win32gui.GetWindowText(hwnd)


# In[ ]:


while True:
    if "Tomb Raider" in get_active_window_title():
        detectObject()
        time.sleep(0.5)
    else:
        print("Not in Tomb raider")
        time.sleep(2)


# In[ ]:





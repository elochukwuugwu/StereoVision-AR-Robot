import sys
import cv2
import numpy as np
import websocket
import math
import socket
from matplotlib import pyplot as plt

net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
with open('Object names.txt','r') as f:
    classes = f.read().splitlines()


cap = cv2.VideoCapture('bowl recognition.mp4')
#img = cv2.imread('go fund image.jpeg')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))
font = cv2.FONT_HERSHEY_PLAIN


while True:
    _, img = cap.read()
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True,
                                 crop=False)  # Performs mean subtraction, scaling and optionally channel swapping
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    boxes = []
    confidences = []
    class_ids = []
    # print(layerOutputs)
    for output in layerOutputs:
        for detection in output:  # iterate thru the nested array of layeroutputs to extract data
            scores = detection[5:]
            class_id = np.argmax(scores)  # finds the largest score to get the computer's best guess
            confidence = scores[class_id]
            if confidence > 0.5:  # bound box is first 4 values of 'detection'
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)  # width and height

                x = int(center_x - w / 2)  # finding the top left corner
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # performs non maximum suppression to prevent unwanted bounding boxes
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + "" + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

    out.write(img)
    cv2.imshow('image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break



cv2.destroyAllWindows()
cap.release()






# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
import cv2
import numpy as np
import websocket
import math
import socket
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
tracker = cv2.legacy_TrackerMOSSE.create()
#tracker = cv2.legacy_TrackerCSRT.create()
_, img = cap.read()
bbox = cv2.selectROI("Tracking", img,False)
tracker.init(img,bbox)
def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255, 0, 255), 3, 1)
    cv2.putText(img, 'Tracking', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
while True:
    timer = cv2.getTickCount()
    ret, img = cap.read()
    ret, bbox = tracker.update(img)
    if ret:
        drawBox(img,bbox)
    else:
        cv2.putText(img,'lost',(75,75),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(img,str(int(fps)),(75,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
    cv2.imshow('Tracking',img)
    if cv2.waitKey(1) == 27:
        break
cap.release()


import sys
import cv2
import numpy as np
import websocket
import math
import socket


def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open('Object names.txt', 'r') as f:
        classes = f.read().splitlines()
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            print(scores)
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    cv2.imshow("Image", img)


def load_image(img_path):
    # image loading
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels


def image_detect(img_path):
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, colors, class_ids, classes, image)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break


def webcam_detect():
    model, classes, colors, output_layers = load_yolo()
    cap = start_webcam()
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()


def start_video(video_path):
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()

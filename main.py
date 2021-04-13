#import sys
import cv2
import numpy as np
#import websocket
#import math
#import socket
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from matplotlib import pyplot as plt


def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    np.set_printoptions(threshold=np.inf)
    global trackStatus
    return height, width, channels, img


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


def trackprep(boxes, confs):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    bboxes = []
    for i in range(len(boxes)):
        if i in indexes:
            bboxes.append(boxes[i])

    return bboxes


def trackObj(boxes, img, ):
    trackers = cv2.legacy_MultiTracker.create()
    algo = cv2.legacy_TrackerCSRT.create()
    for box in boxes:
        trackers.add(algo, img, tuple(box))
    trackstat = True
    while True:
        timer = cv2.getTickCount()
        ret, boxes = trackers.update(img)
        if ret:
            drawBoxes(img, boxes)
        else:
            cv2.putText(img, 'lost', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if cv2.waitKey(2000) == 27:
                break
            trackstat = False
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(img, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Tracking', img)
        if cv2.waitKey(1) == 27:
            break
    return trackstat


def drawBoxes(img, boxes):
    for bbox in boxes:
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
        cv2.putText(img, 'Tracking', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def detntrack(height, width, img):
    net, classes, colors, outputlayers = load_yolo()
    trackStatus = False
    while not trackStatus:
        blob, outputs = detect_objects(img, net, outputlayers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        boxes = trackprep(boxes, confs)
        trackStatus = trackObj(boxes, img)


vertices = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
)

edges = (
    (0, 1),
    (0, 3),
    (0, 4),
    (2, 1),
    (2, 3),
    (2, 7),
    (6, 3),
    (6, 4),
    (6, 7),
    (5, 1),
    (5, 4),
    (5, 7)
)


def Cube():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()


def rotate_cube():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    glTranslatef(0.0, 0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        Cube()
        pygame.display.flip()
        pygame.time.wait(10)


def load_image(imgPath):
    img_col = cv2.imread(imgPath)
    img_gray = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)
    return img_col, img_gray


def shi_tomasi_det(img_gray):
    corners = cv2.goodFeaturesToTrack(img_gray, 25, 0.01, 10)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img_gray, (x, y), 3, 255, -1)

    plt.imshow(img_gray), plt.show()


def create_disp_map(imgL_path, imgR_path, scale_percent):
    imgL = cv2.imread(imgL_path)
    imgR = cv2.imread(imgR_path)
    print('files read')
    width = int(imgL.shape[1] * scale_percent / 100)
    height = int(imgL.shape[0] * scale_percent / 100)
    dim = (width, height)
    imgL = cv2.resize(imgL, dim)
    imgR = cv2.resize(imgR, dim)
    print('files scaled')
    imgL = cv2.fastNlMeansDenoisingColored(imgL, None, 10, 10, 7, 21)
    imgR = cv2.fastNlMeansDenoisingColored(imgR, None, 10, 10, 7, 21)
    print('Denoising complete')
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    print('converted to Gray')
    #stereo = cv2.StereoBM_create(numDisparities=96, blockSize=11)
    stereo = cv2.StereoSGBM_create(minDisparity=0,numDisparities=64,blockSize=11,P1=0,
                                   P2=0,disp12MaxDiff=0,preFilterCap=0,uniquenessRatio=0,speckleWindowSize=0,
                                   speckleRange=0,mode=0
                                 )
    disparity = stereo.compute(imgL_gray, imgR_gray)
    print('Disparity Map generated')
    return disparity, stereo, imgL_gray, imgR_gray, imgL, imgR


def point_cloud(disparity, stereo, imgL_gray, imgR_gray):
    rev_proj_matrix = np.zeros((4, 4))
    cv2.stereoRectify(cameraMatrix1=None, cameraMatrix2=None, distCoeffs1=0, distCoeffs2=0,
                      imageSize=imgL_gray.shape[:2], R=np.identity(3), T=np.array([0.54, 0., 0.1]),
                      R1=None, R2=None, P1=None, P2=None, Q=rev_proj_matrix)
    points = cv2.reprojectImageTo3D(disparity, rev_proj_matrix)
    return points, rev_proj_matrix


disp, stereo, grayL, grayR, colL, colR = create_disp_map( 'Left-Pic.jpg','Right-Pic.jpg',30)
while True:
    cv2.imshow('Right', grayR)
    cv2.imshow('Left', grayL)
    cv2.imshow('disp', disp)
    if cv2.waitKey(100) == 27:
        cv2.destroyAllWindows()
        break

plt.imshow(disp, 'gray')
plt.show()

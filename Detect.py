from asyncio import FastChildWatcher
import cv2
import numpy as np
import argparse

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def yellow_detect(img):
    frame_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(frame_HSV, (20, 28, 20), (33, 255, 255))
    return frame_threshold

def closing(img):
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing

def dilation(img):
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(img, kernel, iterations = 7)
    return dilation

def findObject(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(img.shape[:2], dtype = "uint8")
    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        cv2.drawContours(mask, [c], -1, 255, -1)
        x,y,w,h = cv2.boundingRect(c)
    fillimg = cv2.bitwise_and(imgorg, imgorg, mask = mask)
    return fillimg

def gradient(img):
    kernel = np.ones((9,9),np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    gradient = cv2.cvtColor(gradient, cv2.COLOR_BGR2GRAY)
    return gradient

def minium(k):
    if (k > 40):
        k = float(k)
        k /= 6
        return np.uint32(k * 5)
    elif k < 35:
        k /= 9
        return np.uint32(k * 10)
    return np.uint32(k)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
help = "Path to the image")
args = vars(ap.parse_args())
img = cv2.imread(args["image"])
imgorg = img

img_Lighter = change_brightness(img, value = -30)     

img_Yellow_detect = yellow_detect(img_Lighter)

img_Closed = closing(img_Yellow_detect)

img_Dilated = dilation(img_Closed)

masked = findObject(img_Dilated)

masked_hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)

img_Gradient = gradient(masked_hsv)

detected_circles = cv2.HoughCircles(img_Gradient, 
                   cv2.HOUGH_GRADIENT, 1, 50, param1 = 50,param2 = 30, minRadius = 25, maxRadius = 50)

if detected_circles is not None:

    detected_circles = np.uint16(np.around(detected_circles))

    for pt in detected_circles[0, :]:                                       
        a, b, r = pt[0], pt[1], pt[2]
        cv2.circle(imgorg, (a, b), minium(r), (0, 255, 0), 2)

text = "Total detected: " + str(len(detected_circles[0]))
coordinates = (778,708)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255,0,255)
thickness = 2
image = cv2.putText(imgorg, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
cv2.imwrite("Result.jpg", image)

cv2.waitKey(0)
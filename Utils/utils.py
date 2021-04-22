# Python imports
import cv2
import numpy as np 
np.random.seed(200)

# Global Variable
COLORS = dict()
        
def __add_color(detection):
    if detection["label"] not in COLORS.keys():
        COLORS[detection["label"]] = np.random.uniform(0, 255, size=(1, 3)).flatten()
        
def get_videotimestamp(cameraCapture):
    """
    Function to get the timestamps of the video. 

    INPUT
        cameraCapture(<class 'cv2.VideoCapture'>):    Video capture object for the video currenty read. 

    RETURN
        <str>
        Current timestamp of the video in format H:M:S.MS
    """
    seconds = 0
    minutes = 0
    hours = 0
    milliseconds = cameraCapture.get(cv2.CAP_PROP_POS_MSEC)
    seconds = milliseconds//1000
    milliseconds = milliseconds%1000
    if seconds >= 60:
        minutes = seconds//60
        seconds = seconds % 60
    if minutes >= 60:
        hours = minutes//60
        minutes = minutes % 60
    return "{}:{}:{}.{}".format(int(hours), int(minutes), int(seconds), int(milliseconds))

def pp_detectionlist(dectList):
    """
    Function to pretty print detection from single image.

    INPUT
        dectList(list): Output the list of dictonary with {label, confidence, box}
    """
    for detection in dectList:
        (startX, startY, endX, endY) = detection["bbox"]
        obj = detection["label"]
        confidence = detection["confidence"]
        print("[DETECTED] {}: {:.2f}".format(obj, confidence))

def labelImage(image, dectList):
    """
    Function to label the image with detection bouding box and confidence. 
    Label information on the image is "inplace" operation. 

    INPUT
        image(numpy.ndarray):   Image on which labels are to put. 
        dectList(list):         Output the list of dictonary with {label, confidence, box}
    """
    for detection in dectList:
        # Empty string for the label
        label = str()   
        # Add a new color to the dict if we have not seen the label before
        __add_color(detection)
        
        (startX, startY, endX, endY) = detection["bbox"]
        obj = detection["label"]
        confidence = detection["confidence"]

        # Rectangle around the objects detected
        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[obj], 2)
        # Label on the object
        y = startY - 15 if startY - 15 > 15 else startY + 15
        label = "{}: {:.2f}%".format(obj, confidence)
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[obj], 2)
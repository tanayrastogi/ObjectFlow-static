# Python Imports
import numpy as np
import cv2
import time 
from shapely import geometry
import argparse
import imutils

# Local libraries
import ObjectDetection 
from pyimagesearch.centroidtracker import CentroidTracker
import Utils.utils as ut



# Check if the points are in the polygon
def check_box_in_polygon(polygon, bbox):
    """
    Function to check if the point is in a polygon or not
    """
    # Point to check for the box
    point = ((bbox[0] + bbox[2])/2, bbox[3])
    
    # Polygon
    poly = geometry.Polygon(polygon)
    # Point
    pt = geometry.Point(point)
    # Return if polygon conatins the point
    return poly.contains(pt)


def detections_in_polygon(POLYGON, detections):
    return [d for d in detections if check_box_in_polygon(POLYGON, d["bbox"])]

def update_trail_colors(objects):
    global trail_color
    for key in objects:
        if key not in trail_color.keys():
            trail_color[key] = np.random.uniform(0, 255, size=(1, 3))
    return trail_color

def draw_object_on_image(frame, objects):
    global trail_color
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, trail_color[objectID].flatten(), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, trail_color[objectID].flatten(), -1)

def update_footprints(objects):
    global footprints
    global trail_color
    for (objectID, centroid) in objects.items():
        # Add to the footprint dict
        footprints.append(dict(location=centroid,
                               color=trail_color[objectID].flatten()))

def draw_footprint_on_image(frame):
    global footprints
    # Draw all the foot prints on the image
    for foot in footprints:
        cv2.circle(frame, (foot["location"][0], foot["location"][1]),
                   2, foot["color"], -1)

def draw_polygon(frame, pts):
    bbox = np.array(pts, np.int32)
    bbox = bbox.reshape((-1,1,2))
    cv2.polylines(frame, [bbox], True, (0, 0, 255), 1)

if __name__ == "__main__":

    # Create CL Arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--video", required=True,
        help="path to the video.")
    ap.add_argument("-p", "--polygon", nargs="+", default=None,
        help="List of points of the polygon within objects are tracked.")
    ap.add_argument("-d", "--maxdisp", type=int, default=30,
        help="Max number of frame to consider for object to disappear.")
    ap.add_argument("-s", "--skipframes", type=int, default=1,
        help="number of frames to skip for Object Detection.")
    ap.add_argument("-c", "--classes", nargs="+", default=["person"],
        help="Classes to detect in obejct detection.")
    ap.add_argument("--confidence", type=float, default=0.6,
        help="Base confidence for object detection.")
    args = vars(ap.parse_args())

    # # ENVIRONMENT VARIABLES # #
    VIDEO = args["video"]
    FRAMES_TO_SKIP = args["skipframes"] # Frame
    MAXDISP = args["maxdisp"]
    CLASSES_TO_DETECT = args["classes"]
    BASE_CONFIDENCE = args["confidence"]
    # Bounding points for the polygon
    POLYGON = [tuple(int(i) for i in args["polygon"][i*2:i*2+2]) for i in range(4)]

    # # INITIALIZTION # #
    # Variables
    totalFrames = 0         # Variable to keep track of the frames
    trail_color = dict()    # Variable to keep track of all the trails for all object detected 
    footprints  = list()    # Varibale to keep track of footpritns for objects

    # Object Detection Class
    modelname = "faster_rcnn_inception_v2_coco_2018_01_28"
    proto     = modelname+".pbtxt"
    classes   = "object_detection_classes_coco.txt"
    graph     = "frozen_inference_graph.pb"
    obd = ObjectDetection.TensorflowModel(modelname, proto, graph, classes,
                                BASE_CONFIDENCE, CLASSES_TO_DETECT)

    # Video Sequence
    vs = cv2.VideoCapture(VIDEO)

    # Object Tracker 
    # Consecutive frames that an object will be considered dissappeared
    ct = CentroidTracker(maxDisappeared=MAXDISP)

    # Start Reading the frames video sequence
    while True:
        # Read frames
        ret, frame = vs.read()
        # check to see if we have reached to end of stream
        if not ret:
            break

        # Reshape the frame for working on it and take height, width
        frame = imutils.resize(frame, width=1280, inter=cv2.INTER_AREA)
        (height, width, channel) = frame.shape
        
        # Get timestamp of the video
        curr_timestamp = ut.get_videotimestamp(vs)

        # Skip frames
        if totalFrames % FRAMES_TO_SKIP == 0:
            
            ####################
            # Object Detection #
            ####################
            detections = obd.detect(frame, imgName=curr_timestamp)
            # Keep only the points that are in the polygon
            if POLYGON is not None:
                detections = detections_in_polygon(POLYGON, detections)
            # Print detected objects
            ut.pp_detectionlist(detections)
            # Label the image with the detected object
            ut.labelImage(image=frame, dectList=detections)

            ###################
            # Object Tracking #
            ###################
            objects = ct.update([d["bbox"] for d in detections])
            update_trail_colors(objects)
            draw_object_on_image(frame, objects)
            update_footprints(objects)
            draw_footprint_on_image(frame)
            if POLYGON is not None:
                draw_polygon(frame, POLYGON)

            # Show frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            time.sleep(0.01)
            
            # if 'p' is pressed, then pause video
            if key == ord('p'):
                print("[INFO] Paused!! Press any key to release ...")
                cv2.imshow("Frame", frame)
                cv2.waitKey(-1)
            if key == ord('q'):
                # if 'q' is pressed, then quit the video
                print("[INFO] Quiting ...")
                break

        # Count frames
        totalFrames += 1
        
    # Close all the open windows
    print("[INFO] Closing the all window ...", end=" ")
    cv2.destroyAllWindows()
    print("Done!")

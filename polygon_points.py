import argparse
import numpy as np
import cv2
import imutils
import time 


# # GLOBAL VARIABLE # #
# Points for the polygon 
pts = list()


# Function when mouse is pressed
def mouse_click(event, x, y, flag, params):   
    global pts   
    workingFrame = frame.copy()

    # Left Buton CLick. Add points to the list
    if event == cv2.EVENT_LBUTTONDOWN:
        print("\n[CLICK] LEFT Button CLICK")
        print("Adding point: ", (x, y))
        pts.append((x, y))  
    # Right Buton CLick. Remove points to the list
    if event == cv2.EVENT_RBUTTONDOWN:
        print("\n[CLICK] RIGHT Button CLICK")
        point = pts.pop()
        print("Removing point: ", point)

    # Draw on the frame
    if len(pts) > 0:
        cv2.circle(workingFrame, pts[-1], 3, (0, 0, 255), -1)
    if len(pts) > 1:
        for i in range(len(pts) - 1):
            cv2.circle(workingFrame, pts[i], 3, (0, 0, 255), -1)
            cv2.line(img=workingFrame, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=1)
    cv2.imshow("Frame", workingFrame)

def draw_bounding_box(frame, pts):
    bbox = np.array(pts, np.int32)
    bbox = bbox.reshape((-1,1,2))
    cv2.polylines(frame, [bbox], True, (0, 0, 255), 1)


if __name__== "__main__":

    # Create CL Arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--video", required=True,
                    help="path to the video.")
    args = vars(ap.parse_args())
        
    VIDEO = args["video"]
    # Video Sequence
    vs = cv2.VideoCapture(VIDEO)

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

        # Print bounding box if there are enough points
        if len(pts) > 3:
            draw_bounding_box(frame, pts)

        # Show frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        time.sleep(0.01)
        # if 'p' is pressed, then pause video
        if key == ord('p'):
            print("\n[INFO] Paused!! Press any key to release ...")
            cv2.imshow("Frame", frame)
            cv2.setMouseCallback("Frame", mouse_click)
            cv2.waitKey(-1)
        if key == ord('q'):
            # if 'q' is pressed, then quit the video
            print("\n[INFO] Quiting ...")
            break

    # Close all the open windows
    print("\n[INFO] Closing the all window ...", end=" ")
    cv2.destroyAllWindows()
    print("Done!")
    print("\n[INFO] The bounding box points are: ")
    print(pts)

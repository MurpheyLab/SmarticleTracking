
import sys
sys.path.append("../")

import cv2
import numpy as np
from apriltag import apriltag
from pdb import set_trace as bp


# Set aspect ratio
# frame_width=2.5*4096/4
# frame_height=2.5*2160/4

frame_width=1080
frame_height=720

# Live camera feed or prerecorded video
from_camera=0
from_video='camera_test.avi'

cap = cv2.VideoCapture(from_video)
cap.set(6, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width,frame_height)) # setting MJPG codec
cap.set(3, frame_width) # Width
cap.set(4, frame_height) # Height


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector = apriltag("tagStandard41h12")
    detections = detector.detect(gray)

    if len(detections):
        for i in range(len(detections)):
        # cv2.circle(img,detections[0]['center'],1,(0,255,0))
            center=tuple(detections[i]['center'])
            center1=int(center[0])
            center2=int(center[1])
            cv2.circle(gray,(center1,center2),10,(0,0,255))

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()

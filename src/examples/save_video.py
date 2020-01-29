import numpy as np
import cv2
from pdb import set_trace as bp

k = 2
frame_height=720
frame_width=1280#int(frame_height*1.77778)
# frame_width=1280
# frame_height=720
fps=20

from_camera=0

cap = cv2.VideoCapture(from_camera)
cap.set(6, cv2.VideoWriter_fourcc(*'MJPG')) # setting MJPG codec
cap.set(3, frame_width) # Width
cap.set(4, frame_height) # Height
cap.set(5,fps) # fps
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) #disable autofocus
cap.set(28,0) # set manual focus

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('camera_test.avi',cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width,frame_height))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # bp()

    if ret:
        # Write the frame into the file 'output.avi'
        try:
            out.write(frame)
            # Our operations on the frame come here
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # bgr = cv2.cvtColor(frame,0)
            # Display the resulting frame
            x =235
            y =35
            w =835
            h =640
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255))
            roi = frame[y:y+h, x:x+w]
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            pass
    else:
        break

# When everything done, release the capture
cap.release()
out.release()

cv2.destroyAllWindows()

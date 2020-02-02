import sys
sys.path.append("../src")

import cv2
from camera import Camera


# Live camera feed or prerecorded video
video_out_path = 'camera_test3.avi'
fps = 20
frame_height = 720
frame_width = 1280


cam = Camera(save_video = 'camera_test3.avi', frame_height=frame_height,\
    frame_width=frame_width, fps=fps)
while(True):
    cam.capture_frame()
    cam.write_frame()
    cam.show_frame()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cam.close()

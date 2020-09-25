import sys
sys.path.append("../src")


import cv2
from tracking import Tracking
from camera import Camera
import time


# helper function for doing a step of frame capture and tracking
def step_function(camera, tracking, smart_ids):
    camera.capture_frame()
    tracking.detect_frame(camera.roi)
    tracking.save_detections()
    tracking.draw_lines(camera.frame, smart_ids)

    camera.write_frame()
    camera.show_frame()

# Live camera feed or prerecorded video
from_camera=0
from_video='camera_test2.avi'

video_out_path = 'camera_test3.avi'
fps = 20
frame_height = 1080
frame_width = 1920


cam = Camera(video_source=from_camera,save_video = video_out_path,\
    frame_height=frame_height, frame_width=frame_width, fps=fps)


# IDs of smarticles to be tracked--these correspond to IDs of AprilTags
smart_ids = [1,12]
track = Tracking(smart_ids)


show_timer = True
counter=0
track.start(cam)
step = lambda: step_function(cam,track,smart_ids)
while(True):


    t0 = time.time()
    step()
    if track.q_pressed():
        break
    t_elapsed = time.time()-t0
    # prints true tracking rate; this will be different than the specified fps
    # and will depend on the setup of your system
    # (e.g. you can't track at 30Hz in 4K with 40 tags in the frame)
    if show_timer and counter%10:
        print('Period: {}s, Freq: {}Hz'.format(t_elapsed, 1/t_elapsed))
    counter+=1

# When everything done, release the capture
cam.close()

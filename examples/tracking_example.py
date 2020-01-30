
import sys
sys.path.append("../src")

import cv2
import numpy as np
from apriltag import apriltag
from smarticle_tracking import Tracking
import time

# set to true to print tracking rate (Hz) to screen
show_timer = True

# IDs of smarticles to be tracked--these correspond to IDs of AprilTags
smart_ids = [1,12]

# Live camera feed or prerecorded video
from_camera=0
from_video='camera_test2.avi'

track = Tracking(smart_ids, video_source=from_camera, frame_width=1280,\
 frame_height=720, fps=24, save_video=None, show_video=True)
t0 = time.time()
counter = 0
while(True):
    # tracking step
    t0 = time.time()
    track.step()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    t_elapsed = time.time()-t0
    # prints true tracking rate; this will be different than the specified fps
    # and will depend on the setup of your system
    # (e.g. you can't track at 30Hz in 4K with 40 tags in the frame)
    if show_timer and counter%10:
        print('Period: {}s, Freq: {}Hz'.format(t_elapsed, 1/t_elapsed))
    counter+=1

# When everything done, release the capture
track.close_camera()

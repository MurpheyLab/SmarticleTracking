
import sys
sys.path.append("../src")

import cv2
import numpy as np
from apriltag import apriltag
from tracking import Tracking
from tracking_object import TrackingObject
import time

# set to true to print tracking rate (Hz) to screen
show_timer = True
fps = 20

# IDs of smarticles to be tracked--these correspond to IDs of AprilTags
smart_ids = [1,12]

ring_ids = [100, 101, 102]

tag_ids = smart_ids+ring_ids

def get_ring_center(trck, ring_ids):
    ring_tag_xy = [obj.x[:2] for obj in trck.tracking_objects if obj.id in ring_ids]
    ring_center = sum(ring_tag_xy)/len(ring_tag_xy)
    r = 3
    color = (0,255,255)
    thick = -1
    cv2.circle(trck.frame, (int(ring_center[0]),int(trck.frame_height-ring_center[1])),\
    r, color, thick)
    cv2.imshow('frame', trck.frame)
    return ring_center

# Live camera feed or prerecorded video
from_camera=0
from_video='camera_test2.avi'

track = Tracking(tag_ids, video_source=from_camera, frame_width=1280,\
 frame_height=720, fps=fps, save_video=None, show_lines=False, show_video=True)
t0 = time.time()
counter = 0
while(True):
    # tracking step
    t0 = time.time()
    track.step()
    get_ring_center(track, ring_ids)
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


import sys
sys.path.append("../src")



import cv2
from tracking import Tracking
from camera import Camera
import time

# constants
SMARTICLE_TAG_LENGTH_MM = 11.2
RING_DIAM_MM = 200
# roi (region of interest) defines square to look for tags
# currently set to square with side length 1.5 times the ring diameter
roi_safety_factor = 1.5


# helper function for doing a step of frame capture and tracking
def step_function(camera, tracking, smart_ids, ring_ids, side_length):
    r = 3
    color1 = (0,255,255)
    color2 = (255,0,0)
    thick1 = -1
    thick2 = 2
    camera.capture_frame()
    tracking.detect_frame(camera.roi)
    tracking.save_detections(offset=camera.roi_dims[:2])
    tracking.draw_lines(camera.frame, smart_ids)
    ring_center = tracking.get_centroid(ring_ids)
    camera.set_roi_dims(ring_center,side_length,side_length)
    cv2.circle(camera.frame, (int(ring_center[0]),int(ring_center[1])),\
    r, color1, thick1)
    xlim = int(camera.roi_dims[0]+camera.roi_dims[2])
    ylim = int(camera.roi_dims[1]+camera.roi_dims[3])
    cv2.rectangle(camera.frame,(int(camera.roi_dims[0]),int(camera.roi_dims[1])),\
        (xlim,ylim), color2, thick2)

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
ring_ids = [100, 101, 102]
tag_ids = smart_ids+ring_ids
values = []
# make length_dict, relating tag_id to tag_dimensions
# smarticle tags get dimension set by constat SMARTICLE_TAG_LENGTH_MM)
for id in tag_ids:
    if id in smart_ids:
        values.append(SMARTICLE_TAG_LENGTH_MM)
    else:
        values.append(None)
    length_dict = dict(zip(tag_ids,values))
track = Tracking(tag_ids, history_len=None, length_dict=length_dict)


show_timer = True
counter=0
track.start(cam)
scale = track.get_scale_factor()
side_length = RING_DIAM_MM*roi_safety_factor*scale
step = lambda: step_function(cam,track,smart_ids, ring_ids, side_length)
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

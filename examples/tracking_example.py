
# import sys
# sys.path.append("../")

import cv2
import numpy as np
from apriltag import apriltag
from pdb import set_trace as bp
from modules.detection import Detection
from modules import tracking
import time

# IDs of smarticles to be tracked--these correspond to IDs of AprilTags
smart_ids = [8,9,10]

# Live camera feed or prerecorded video
from_camera=0
from_video='camera_test2.avi'

track = tracking.Tracking(from_camera, smart_ids, show_video=True)
t0 = time.time()
while(True):
    # tracking step
    track.step()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
track.close()

#tracking.py
# Smarticle tracking w/ April tags
# Created Nov 19, 2019
# Alex Samland (alexsamland@u.northwestern.edu)

import numpy as np
import cv2
from copy import deepcopy
from apriltag import *
import time
from tracking_object import TrackingObject


################################################################################
#                                  Tracking Class                              #
################################################################################

class Tracking(object):
    '''
    ## Description
    ---
    Class for tracking multiple AprilTag objects and managing use of webcam with OpenCV

    **Public Attributes (for the user):**

    * TODO

    **Private Attributes (for the class):**

    * TODO
    '''


    def __init__(self, tag_ids, history_len=None, length_dict=None):
        '''

        ## Arguments
        ---

        | Argument     | Type            | Description                                                                             | Default Value  |
        | :------      | :--             | :---------                                                                              | :-----------   |
        | tag_ids      | `list` of `int` | List of tag IDs to track                                                                | N/A            |
        | frame_width  | `int`           | Frame width of camera capture                                                           | N/A            |
        | frame_height | `int`           | Frame height of camera capture                                                          | N/A            |
        | fps          | `int`           | Frames per second of camera capture                                                     | N/A            |
        | video_source | `string`        | *Optional:* Path of input video file                                                    | 0              |
        | save_video   | `string`        | *Optional:* Save video to specified path                                                | `None`         |
        | show_video   | `bool`          | *Optional:* Show video to screen if `True`                                              | `False`        |
        | history_len  | `int`           | *Optional:* Max length of tracking history to be saved                                  | `None`         |
        | roi_dims     | `list` of `int` | *Optional:* Two element list that specifies offset from detection frame to global frame | `None`         |
        |<img width=300/>|<img width=300/>|<img width=900/>|<img width=250/>|

        '''

        # save as attribute
        self.history_len = history_len

        # Make sure tag_ids are in ascending order
        self.tag_ids = deepcopy(tag_ids)
        self.tag_ids.sort()
        self.line_length = 25

        if length_dict is None:
            values = [None]*len(self.tag_ids)
            self.length_dict = dict(zip(self.tag_ids,values))
        else:
            assert len(length_dict) == len(self.tag_ids),\
             "length_dict should be a dictionary of same length of tag_ids"
            self.length_dict = length_dict

        # April Tag Detector Object, specify tag family
        self.detector = apriltag("tagStandard41h12")

        # initialize tracking objects
        self.tracking_objects = [TrackingObject(tag_id, history_length=self.history_len,\
            tag_length=self.length_dict[tag_id]) for tag_id in self.tag_ids]

    @classmethod
    def q_pressed(self):
        return cv2.waitKey(1) & 0xFF == ord('q')



    def start(self,cam):
        '''Initializes April tag detector and creates tracking objects.
        Additionally gets initial position of objects and sets time for t0'''

        # set t0 for tracking data
        self.t0 = time.time()

        for obj in self.tracking_objects:
            det = None
            t_start = time.time()
            while det is None:
                # capture frame and region of interest, specified by crop region
                cam.capture_frame()
                # detect april tags in frame
                detections = self.detect_frame(cam.frame)
                ids_detected = [x['id'] for x in detections]
                if obj.id not in ids_detected:
                    det = None
                else:
                    det = detections[ids_detected.index(obj.id)]
                if (time.time()-t_start)>5:
                    raise Exception('Tag {}  could not be found in frame'.format(obj.id))

            obj.init_detection(time.time()-t_start,det)
            print('Tag {} detected in frame'.format(obj.id))


    def detect_frame(self,frame):
        '''
        ## Description
        ---
        Returns state (x, y, theta) given detection and offset

        ## Arguments
        ---

        | Argument| Type         | Description              | Default Value  |
        | :------ | :--          | :---------               | :-----------   |
        | frame     | `np.array` | Frame to detect tags in  | N/A            |
        |<img width=300/>|<img width=300/>|<img width=900/>|<img width=250/>|

        ## Returns
        ---
        `list` of `dict`s corresponding to each tag detected
        '''

        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.detections = self.detector.detect(gray)
        return self.detections

    def save_detections(self, detections=None, offset=None):
        '''
        ## Description
        ---
        Saves detection data to TrackingObject data class objects

        ## Arguments
        ---

        | Argument       | Type             | Description                    | Default Value  |
        | :------        | :--              | :---------                     | :-----------   |
        | detections     | `list` of `dict` | List of detection dictionaries | N/A            |
        |<img width=300/>|<img width=300/>|<img width=900/>|<img width=250/>|

        ## Returns
        ---
        void
        '''
        if detections is None:
            detections = self.detections
        if offset is None:
            offset = [0,0]
        t= time.time()-self.t0
        ids_detected = [x['id']for x in detections]
        for obj in self.tracking_objects:
            if obj.id not in ids_detected:
                obj.add_timestep(t, det = None, offset = offset)
            else:
                obj.add_timestep(t, det = detections[ids_detected.index(obj.id)], offset = offset)

    def draw_lines(self, frame, ids):
        for obj in self.tracking_objects:
            if obj.id in ids:
                # draw line showing orientation of tag
                cv2.line(frame, (int(obj.x[0]),int(obj.x[1])),\
                (int(obj.x[0]+self.line_length*np.cos(obj.x[2])), int(obj.x[1]+self.line_length*np.sin(obj.x[2]))),\
                (0,255,0),2)

    def get_centroid(self, tag_ids):
        ring_tag_xy = [obj.x[:2] for obj in self.tracking_objects if obj.id in tag_ids]
        return sum(ring_tag_xy)/len(ring_tag_xy)


    def get_scale_factor(self):
        '''
        Get scale factor (mm/pixels) of camera setup
        '''
        scale_factors = [obj.scale_factor for obj in self.tracking_objects if obj.tag_length is not None]
        # scale_factors = [obj.scale_factor for obj in objects_w_tag_length]
        self.scale_factor = sum(scale_factors)/len(scale_factors)
        print('Scale factor of {} pixels/mm'.format(self.scale_factor))
        self.line_length = 15*self.scale_factor
        return self.scale_factor

    def save_data(self, path, local_copy=False):
        '''
        ## Description
        ---
        Saves data to given file path as .csv file in following format
        (timestamp, state_tag_1, state_tag2,...state_tag_n)

        ## Arguments
        ---

        | Argument   | Type     | Description                                | Default Value  |
        | :------    | :--      | :---------                                 | :-----------   |
        | path       | `string` | Path to save data                          | N/A            |
        | local_copy | `bool`   | *Optional:* Returns data locally if `True` | `False`        |
        |<img width=300/>|<img width=300/>|<img width=900/>|<img width=250/>|

        ## Returns
        ---
        void
        '''
        t = np.array([self.tracking_objects[0].t_history]).T
        header = 'time (s), '
        data = np.hstack([np.array(obj.history) for obj in self.tracking_objects])
        for obj in self.tracking_objects:
            header+= 'x{0}, y{0}, theta{0}, '.format(obj.id)
        data_out = np.hstack([t,data])
        np.savetxt(path,data_out, delimiter=',', header=header)

        if local_copy:
            return t,S

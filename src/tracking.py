#tracking.py
# Smarticle tracking w/ April tags
# Created Nov 19, 2019
# Alex Samland (alexsamland@u.northwestern.edu)

import numpy as np
import cv2
from copy import deepcopy
from apriltag import *
import time


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


    def __init__(self, tag_ids, frame_width, frame_height, fps,\
     video_source=0, save_video=None, show_video=False,\
     history_len=None, roi_dims=None):
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

        # save as attributes
        self.save_video = save_video
        self.show_video = show_video
        self.history_len = history_len
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

        # Make sure tag_ids are in ascending order
        self.tag_ids = deepcopy(tag_ids)
        self.tag_ids.sort()

        self._init_camera(video_source)

        # region of interest parameters: should be 4 element list of the form: [x, y, w, h]
        if roi_dims is not None:
            assert len(roi_dims) is 4, 'roi_dims is 4 element list of form: [x, y, w, h]'
            self.roi_dims = roi_dims
        # if roi_dims is not specified
        else:
            # set roi_dims so that None of the image is cropped
            self.roi_dims = [0,0, self.frame_width, self.frame_height]

        self._init_tracking()


    def _init_camera(self, video_source):
        '''Initializes camera with specified settings as tuned tracking settings
        (ie. turns off autofocus, sets brightness and contrast)'''
        # Camera Settings
        self.cap = cv2.VideoCapture(video_source) # sets input source for video capture
        self.cap.set(6, cv2.VideoWriter_fourcc(*'MJPG')) # setting MJPG codec
        self.cap.set(3, self.frame_width) # Width
        self.cap.set(4, self.frame_height) # Height
        self.cap.set(5,self.fps) # fps
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # disable autofocus
        self.cap.set(28,0) # set manual focus
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS,30) # low brightness
        self.cap.set(cv2.CAP_PROP_CONTRAST,100) # high contrast
        if self.save_video is not None:
            # Save video to file
            self.out = cv2.VideoWriter(save_video,cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (self.frame_width,self.frame_height))
        else:
            self.out = None

    def _init_tracking(self):
        '''Initializes April tag detector and creates tracking objects.
        Additionally gets initial position of objects and sets time for t0'''
        # April Tag Detector Object, specify tag family
        self.detector = apriltag("tagStandard41h12")

        # initialize tracking objects
        self.tracking_objects = [TrackingObject(tag_id, self.frame_height, history_length=self.history_len) for tag_id in self.tag_ids]

        # set t0 for tracking data
        self.t0 = time.time()

        for obj in self.tracking_objects:
            det = None
            t_start = time.time()
            while det is None:
                # capture frame and region of interest, specified by crop region
                [self.frame,self.roi] = self.capture_frame()
                # detect april tags in frame
                detections = self.detect_frame(self.roi)
                ids_detected = [x['id'] for x in detections]
                if obj.id not in ids_detected:
                    det = None
                else:
                    det = detections[ids_detected.index(obj.id)]
                if (time.time()-t_start)>5:
                    print('Tag {}  could not be found in frame'.format(obj.id))
                    break

            obj.init_detection(0,det)
            print('Tag {} detected in frame'.format(obj.id))
            self.scale_factor = self.get_scale_factor()
            print('Scale factor of {} pixels/mm'.format(self.scale_factor))

    def capture_frame(self):
        '''
        ## Description
        ---
        Captures frame with attribute `cap` and crops according to `roi_dims`

        ## Arguments
        ---
        None

        ## Returns
        ---
        3D `np.array` of RGB pixel values for whole frame
        3D `np.array` of RGB pixel values for whole specified region of interest (roi)

        '''
        # region of interest (crop region) dimensions
        [x, y, w, h] = self.roi_dims
        ret, frame = self.cap.read()
        # save cropped frame
        roi = frame[y:y+h, x:x+w]
        return [frame,roi]

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
        return self.detector.detect(gray)

    def save_detections(self, detections):
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

        # x and y offset of crop region
        offset = self.roi_dims[0:2]
        t= time.time()-self.t0
        ids_detected = [x['id']for x in detections]
        for obj in self.tracking_objects:
            if obj.id not in ids_detected:
                obj.add_timestep(t, det = None, offset = offset)
            else:
                obj.add_timestep(t, det = detections[ids_detected.index(obj.id)], offset = offset)
                if self.show_video is True and obj.id<100:
                    # draw line showing orientation of tag
                    cv2.line(self.frame, (int(obj.x[0]),int(self.frame_height-obj.x[1])),\
                    (int(obj.x[0]+25*np.cos(-obj.x[2])), int(self.frame_height-obj.x[1]+25*np.sin(-obj.x[2]))),\
                    (0,255,0),2)


    def get_scale_factor(self):
        '''
        Get scale factor (pixels/mm) of camera setup
        '''
        scale_factors = [obj.scale_factor for obj in self.tracking_objects if obj.tag_length is not None]
        return sum(scale_factors)/len(scale_factors)

    def step(self):
        '''
        ## Description
        ---
        Captures frame, detects tags, and saves detection data to TrackingObjects data class object

        ## Arguments
        ---

        None

        ## Returns
        ---
        void
        '''
        [self.frame, self.roi] = self.capture_frame()
        if self.save_video is True:
            self.out.write(self.frame)

        self.detections = self.detect_frame(self.roi)
        self.save_detections(self.detections)

        if self.show_video is True:
            cv2.imshow('frame', self.frame)

    def close_camera(self):
        '''
        ## Description
        ---
        Closes camera object along with all camera windows open

        ## Arguments
        ---
        None

        ## Returns
        ---
        void
        '''
        self.cap.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()

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

        t = np.array(self.tracking_objects[0].t_history)
        data = np.hstack([np.array(obj.history) for obj in self.tracking_objects])
        S = data[:-1]
        data_out = np.hstack([t,S])
        np.savetxt(path,data_out,delimiter=',')

        if local_copy:
            return t,S

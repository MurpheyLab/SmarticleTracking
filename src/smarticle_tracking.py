#smarticle_tracking.py
# Smarticle tracking w/ April tags
# Created Jan 30, 2020
# Alex Samland (alexsamland@u.northwestern.edu)

import numpy as np
import numpy.matlib
from collections import deque
import cv2
from copy import deepcopy
# from apriltag import *
import time


################################################################################
#                                  TrackingObject Class                        #
################################################################################

class TrackingObject(object):
    '''
    ## Description
    ---
    This class is a data struct for each April Tag in the frame to store its current state
    (x, y, theta) as well as a timestamped history of the state. The class also linearly
    interpolates over missed detections to smooth the data.

    **Public Attributes (for the user):**

    * **id**: April Tag ID (e.g. 1, 3, 16). This corresponds to the smarticle ID
    * **x**: state (x, y, theta) of most recent detection of tag
    * **t**: time of most recent detection of tag
    * **history**: history of states (x,y, theta) of tag
    * **t_history**: history of detection times of tag

    **Private Attributes (for the class):**

    * **_frame_height**: used to flip y axis so that origin is in bottom left corner of frames
    * **_missed frames**: used for keeping track of missed frames that require linear interpolation
    * **_object_detected**: flag for indicating whether tag has been initially detected
    '''

    def __init__(self, ID, frame_height, history_length=None):
        '''
        ## Arguments
        ---

        | Argument         | Type    | Description                           | Default Value  |
        | :------          | :--     | :---------                            | :-----------   |
        | ID               | `int`   | ID of corresponding April Tag         | N/A            |
        | frame_height     | `int`   | Pixel height of frame                 | N/A            |
        | history_length   | `int`   | Optional max history length to record | `None`         |
        |<img width=350/>|<img width=250/>|<img width=800/>|<img width=500/>|

        '''

        # attributes to be accessed by user (Public)
        self.id = ID
        self.x = np.zeros(3)
        self.t = 0
        # use a deque data structure for history
        # set a max length of 150 elements or about 10s of data
        self.history = deque(maxlen=history_length)
        self.t_history = deque(maxlen=history_length)

        # attributes to be used within class (Private)
        self._frame_height = frame_height;
        self._missed_frames = 0
        self._object_detected = False

    def _get_state(det, offset=None):
        '''
        ## Description
        ---
        Returns state (x, y, theta) given detection and offset

        ## Arguments
        ---

        | Argument| Type           | Description                                                                 | Default Value  |
        | :------ | :--            | :---------                                                                  | :-----------   |
        | det     | `dict`         | Detection of tag from AprilTag library                                      | N/A            |
        | offset  | `list` of `int`| Two element list that specifies offset from detection frame to global frame | `None`         |
        |<img width=250/>|<img width=350/>|<img width=1400/>|<img width=250/>|

        ## Returns
        ---
        `np.array` state [x, y, theta]
        '''
        # x and y position of center of tag
        center = det['center']
        # flip y axis from camera to put origin in bottom left corner
        center[1]=self._frame_height-center[1]
        # point at center top of tag is average of top right and top left corners
        center_top = (det['lb-rb-rt-lt'][2]+det['lb-rb-rt-lt'][3])/2.
        center_top[1]=self._frame_height-center_top[1]
        # calculate theta as angle between center and cetner top and angle wrap so it is between 0 and 2pi
        theta = np.mod(np.arctan2(center_top[1]-center[1],center_top[0]-center[0]),2*np.pi)
        # record angle so that there are no discontinuities (prevent wrapping between 0 and 2*pi)
        dtheta = theta-self.x[2]
        dtheta = np.mod((np.pi+dtheta),2*np.pi)-np.pi
        new_theta = self.x[2]+dtheta
        return np.append(center+offset,new_theta)

    def _smooth_missed_frames():
        '''
        ## Description
        ---
        smooths data by linearly interpolating over missed frames

        ## Arguments
        ---
        N/A

        ## Returns
        ---
        void
        '''
        # time of last sucessful detection before this most recent one
        t0 = self.t_history[-(1+self._missed_frames)]
        # calculate slope between points
        m = (self.x-self.history[-(1+self._missed_frames)])/(self.t-t0)
        # iterate through all of the missed frames
        while self._missed_frames > 0:
            # calculate dt between missed frame and t0
            dt = self.t_history[-(self._missed_frames)] - t0
            # apply linear smoothing
            self.history[-(self._missed_frames)] += m*dt
            # move to next missed frame
            self._missed_frames-=1


    def init_detection(self, t, det, offset = None):
        '''
        ## Description
        ---
        Initializes detections of tags with initial state and time of tag at first detection

        ## Arguments
        ---

        | Argument| Type           | Description                                                                 | Default Value  |
        | :------ | :--            | :---------                                                                  | :-----------   |
        | t       | `int`          | Time of detection                                                           | N/A            |
        | det     | `dict`         | Detection of tag from AprilTag library                                      | N/A            |
        | offset  | `list` of `int`| Two element list that specifies offset from detection frame to global frame | `None`         |
        |<img width=250/>|<img width=350/>|<img width=1400/>|<img width=250/>|

        ## Returns
        ---
        void
        '''
        self.x = self._get_state(det,offset)
        self.t = t
        # add initial pose and time to history
        self.t_history.append(self.t)
        self.history.append(self.x)
        # set detection flag to true
        self._object_detected = True

    def add_timestep(self, t, det=None, offset = None):
        '''
        ## Description
        ---
        Updates current state and history lists based on provided detection and also handles linear interpolation for missing detections

        ## Arguments
        ---

        | Argument| Type           | Description                                                                 | Default Value  |
        | :------ | :--            | :---------                                                                  | :-----------   |
        | t       | `int`          | Time of detection                                                           | N/A            |
        | det     | `dict`         | Detection of tag from AprilTag library                                      | N/A            |
        | offset  | `list` of `int`| Two element list that specifies offset from global frame to detection frame | `None`         |
        |<img width=250/>|<img width=350/>|<img width=1400/>|<img width=250/>|

        ## Returns
        ---
        void
        '''
        assert self._object_detected is True, "Object not initially detected"

        self.t = t
        # if offset is not provided, set offset to [0, 00] (no offset)
        if offset is None:
            offset = np.array([0,0])

        # if object is not detected in this time frame carry over last beleif of
        # object state and increment the missed frames counter
        if det is None:
            self._missed_frames +=1
        else:
            self.x = self._get_state(det,offset)
            # linearly smooth missed frames
            if self._missed_frames > 0:
                self._smooth_missed_frames()
            # add most recent time step to history
        self.history.append(self.x)
        self.t_history.append(self.t)



################################################################################
#                                  Tracking Class                              #
################################################################################

class Tracking(object):
    '''
    Class for tracking multiple AprilTag objects and managing use of webcam with OpenCV
    '''


    def __init__(self, tag_ids, frame_width, frame_height, fps,\
     video_source=0, save_video=None, show_video=False,\
     history_len=None, roi_dims=None):
        '''
        ## Description
        ---
        Initializes detections of tags with initial state and time of tag at first detection

        ## Arguments
        ---

        | Argument     | Type            | Description                                                                            | Default Value  |
        | :------      | :--             | :---------                                                                             | :-----------   |
        | tag_ids      | `list` of `int` | List of tag IDs to track                                                               | N/A            |
        | frame_width  | `int`           | Frame width of camera capture                                                          | N/A            |
        | frame_height | `int`           | Frame height of camera capture                                                         | N/A            |
        | fps          | `int`           | Frames per second of camera capture                                                    | N/A            |
        | video_source | `string`        | (Optional) Path of input video file                                                    | 0              |
        | save_video   | `string`        | (Optional) Save video to specified path                                                | `None`         |
        | show_video   | `bool`          | (Optional) Show video to screen if `True`                                              | `False`        |
        | history_len  | `int`           | (Optional) Max length of tracking history to be saved                                  | `None`         |
        | roi_dims     | `list` of `int` | (Optional) Two element list that specifies offset from detection frame to global frame | `None`         |
        |<img width=250/>|<img width=350/>|<img width=1400/>|<img width=250/>|

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

        self.init_camera(video_source)

        # region of interest parameters: should be 4 element list of the form: [x, y, w, h]
        if roi_dims is not None:
            assert len(roi_dims) is 4, 'roi_dims is 4 element list of form: [x, y, w, h]'
            self.roi_dims = roi_dims
        # if roi_dims is not specified
        else:
            # set roi_dims so that none of the image is cropped
            self.roi_dims = [0,0, self.frame_width, self.frame_height]

        self.init_tracking()


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
                if (time.time()-t_start)>5:
                    print('Smarticle {}  could not be found in frame'.format(obj.id))
                    break

            det = detections[ids_detected.index(obj.id)]
            obj.init_detection(det)
            print('Tag {} detected in frame'.format(obj.id))

    def capture_frame(self):
        '''
        ## Description
        ---
        Captures frame with attribute `cap` and crops according to `roi_dims`

        ## Arguments
        ---
        N/A

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
        |<img width=250/>|<img width=350/>|<img width=1400/>|<img width=250/>|

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
        |<img width=250/>|<img width=350/>|<img width=1400/>|<img width=250/>|

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
                if self.show_video is True:
                    # draw line showing orientation of tag
                    cv2.line(self.frame, (int(obj.x[0]),int(obj.frame_height-obj.x[1])),\
                    (int(obj.x[0]+25*np.cos(-obj.x[2])), int(obj.frame_height-obj.x[1]+25*np.sin(-obj.x[2]))),\
                    (0,255,0),5)


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

        | Argument   | Type     | Description                               | Default Value  |
        | :------    | :--      | :---------                                | :-----------   |
        | path       | `string` | Path to save data                         | N/A            |
        | local_copy | `bool`   | (Optional) Returns data locally if `True` | `False`        |
        |<img width=250/>|<img width=350/>|<img width=1400/>|<img width=250/>|

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
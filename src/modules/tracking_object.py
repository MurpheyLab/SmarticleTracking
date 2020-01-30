# detection.py
# Smarticle tracking w/ April tags
# Created Nov. 12, 2019
# Alex Samland (alexsamland@u.northwestern.edu)

import numpy as np
import numpy.matlib
from collections import deque

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

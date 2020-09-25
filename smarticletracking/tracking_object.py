#tracking_object.py
# Smarticle tracking w/ April tags
# Created Nov 19, 2109
# Alex Samland (alexsamland@u.northwestern.edu)

import numpy as np
from collections import deque
from copy import deepcopy


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

    * **_missed frames**: used for keeping track of missed frames that require linear interpolation
    * **_object_detected**: flag for indicating whether tag has been initially detected
    '''

    def __init__(self, ID, history_length=None, tag_length=None):
        '''
        ## Arguments
        ---

        | Argument         | Type    | Description                           | Default Value  |
        | :------          | :--     | :---------                            | :-----------   |
        | ID               | `int`   | ID of corresponding April Tag         | N/A            |
        | history_length   | `int`   | Optional max history length to record | `None`         |
        |<img width=300/>|<img width=300/>|<img width=900/>|<img width=250/>|

        '''

        # attributes to be accessed by user (Public)
        self.id = ID
        # if object is a smarticle (ID <100), it has a known tag
        # size which can be used for scaling
        self.tag_length = tag_length
        self.x = np.zeros(3)
        self.t = 0
        # use a deque data structure for history
        # set a max length specified by input
        self.history = deque(maxlen=history_length)
        self.t_history = deque(maxlen=history_length)
        self.scale_factor = None

        # attributes to be used within class (Private)
        self._missed_frames = 0
        self._object_detected = False

    def _get_state(self, det, offset):
        '''
        ## Description
        ---
        Returns state (x, y, theta) given detection and offset

        ## Arguments
        ---

        | Argument| Type       | Description                                                                 | Default Value  |
        | :------ | :--        | :---------                                                                  | :-----------   |
        | det     | `dict`     | Detection of tag from AprilTag library                                      | N/A            |
        | offset  | `np.array` | Two element list that specifies offset from detection frame to global frame | N/A            |
        |<img width=300/>|<img width=300/>|<img width=900/>|<img width=250/>|

        ## Returns
        ---
        `np.array` state [x, y, theta]
        '''
        # x and y position of center of tag
        center = det['center']
        # flip y axis from camera to put origin in bottom left corner
        center[1]=center[1]
        # point at center top of tag is average of top right and top left corners
        center_top = (det['lb-rb-rt-lt'][2]+det['lb-rb-rt-lt'][3])/2.
        center_top[1]=center_top[1]
        # calculate theta as angle between center and cetner top and angle wrap so it is between 0 and 2pi
        theta = np.mod(np.arctan2(center_top[1]-center[1],center_top[0]-center[0]),2*np.pi)
        # record angle so that there are no discontinuities (prevent wrapping between 0 and 2*pi)
        dtheta = theta-self.x[2]
        dtheta = np.mod((np.pi+dtheta),2*np.pi)-np.pi
        new_theta = self.x[2]+dtheta
        return np.append(center+offset,new_theta)

    def _get_scale_factor(self,det):
        '''
        DOC
        '''
        assert self.tag_length is not None, 'Tag length must be specified to get scale factor'
        bottom_left = det['lb-rb-rt-lt'][0]
        bottom_right = det['lb-rb-rt-lt'][1]
        top_right = det['lb-rb-rt-lt'][2]
        top_left = det['lb-rb-rt-lt'][3]
        diag_pixel = 0.5*(np.linalg.norm(bottom_left-top_right)+np.linalg.norm(bottom_right-top_left))
        diag_len = np.sqrt(2)*self.tag_length
        # scale factor pixel/unit mm
        return diag_pixel/diag_len

    def _smooth_missed_frames(self):
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
        |<img width=300/>|<img width=300/>|<img width=900/>|<img width=250/>|

        ## Returns
        ---
        void
        '''
        if offset is None:
            offset = np.zeros(2)
        self.x = self._get_state(det,offset)
        if self.tag_length is not None:
            self.scale_factor = self._get_scale_factor(det)
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
        |<img width=300/>|<img width=300/>|<img width=900/>|<img width=250/>|

        ## Returns
        ---
        void
        '''
        assert self._object_detected is True, "Object not initially detected"

        self.t = t
        # if offset is not provided, set offset to [0, 00] (no offset)
        if offset is None:
            offset = np.zeros(2)

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

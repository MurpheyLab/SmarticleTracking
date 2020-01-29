# detection.py
# Smarticle tracking w/ April tags
# Created Nov. 12, 2019
# Alex Samland (alexsamland@u.northwestern.edu)

import numpy as np
import numpy.matlib
from collections import deque

class Detection(object):
    """This class is basically a data struct for each 'detection' or April Tag in the frame.
    This corresponds to a smarticle, and this class stores the position data of the Smarticle
    with the given ID in time.
    Its current position can be accessed as Detection.x
    A list of its timestamped history of positions is stored in Detection.history"""

    def __init__(self, ID, frame_height, history_length=None):
        '''ID specifies ID of April Tag,'''
        self.id = ID
        self.frame_height = frame_height;
        # use a deque data structure for history
        # set a max length of 150 elements or about 10s of data
        self.history = deque(maxlen=history_length)
        self.t_history = deque(maxlen=history_length)
        # initialize time to zero
        self.t = 0
        self.x = np.zeros(3)
        self.missed_frames = 0
        self.init_frames = 50
        self.object_detected = False

    def is_detected(self,t, det, offset = None):
        # if no offset is provided, set offset equal to zero
        if offset is None:
            offset = [0,0]

        # if object is not detected return false
        if det is None:
            return False

        # else if object is detected, initialize object
        else:
            # set detection flag to true
            self.object_detected = True
            # x and y position of center of tag
            center = det['center']
#             flip y axis from camera to put origin in bottom left corner
            center[1]=self.frame_height-center[1]
            # point at center top of tag is average of top right and top left corners
            center_top = (det['lb-rb-rt-lt'][2]+det['lb-rb-rt-lt'][3])/2.
            center_top[1]=self.frame_height-center_top[1]
            # calculate theta as angle between center and cetner top and angle wrap so it is between 0 and 2pi
            theta = np.mod(np.arctan2(center_top[1]-center[1],center_top[0]-center[0]),2*np.pi)

            # add offset to center position
            self.x = np.append(center+offset,theta)
            self.t = t
            # add initial pose and time to history
            self.t_history.append(self.t)
            self.history.append(np.array([self.x[0],self.x[1],self.x[2]]))
            return True

    def add_timestep(self, t, det=None, offset = None):
        '''Add timestep of data to history list and update current values of position'''
        assert self.object_detected is True, "Object not initially detected"

        self.t = t

        # if offset is not provided, set offset to [0, 00] (no offset)
        if offset is None:
            offset = [0,0]

        # if object is not detected in this time frame carry over last beleif of
        # object state and increment thhe missed frames counter
        if det is None:
            self.t_history.append(t)
            self.history.append(np.array([self.x[0],self.x[1],self.x[2]]))
            self.missed_frames +=1
        else:


            # x and y position of center of tag
            center = det['center']
#             flip y axis from camera to put origin in bottom left corner
            center[1]=self.frame_height-center[1]
            # point at center top of tag is average of top right and top left corners
            center_top = (det['lb-rb-rt-lt'][2]+det['lb-rb-rt-lt'][3])/2.
            center_top[1]=self.frame_height-center_top[1]
            # calculate theta as angle between center and cetner top and angle wrap so it is between 0 and 2pi
            theta = np.mod(np.arctan2(center_top[1]-center[1],center_top[0]-center[0]),2*np.pi)
            dtheta = theta-self.x[2]
            dtheta = np.mod((np.pi+dtheta),2*np.pi)-np.pi
            new_theta = self.x[2]+dtheta
            self.x = np.append(center+offset,new_theta)
            # linearly smooth missed frames
            if self.missed_frames > 0 and (t>2):
                # time of last detection before this most recent one
                t0 = self.t_history[-(1+self.missed_frames)]
                # calculate slope between points
                m = (self.x-self.history[-(1+self.missed_frames)])/(self.t-t0)
                # iterate through all of the missed frames
                while self.missed_frames > 0:
                    dt = self.t_history[-(self.missed_frames)] - t0
                    # apply linear smoothing
                    self.history[-(self.missed_frames)] += m*dt
                    # move to next missed frame
                    self.missed_frames-=1
            # add most recent time step to history
            self.history.append(np.array([self.x[0],self.x[1],self.x[2]]))
            self.t_history.append(self.t)

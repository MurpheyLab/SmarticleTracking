# tracking.py
# Smarticle tracking w/ April tags
# Created Nov. 12, 2019
# Alex Samland (alexsamland@u.northwestern.edu)
import cv2
from copy import deepcopy
from apriltag import *
from modules.detection import *
import time


class Tracking(object):


    def __init__(self,video_source, smart_ids, save_video = None,\
    show_video = False, frame_width=1280, frame_height=720, fps = 4,\
    history_len = None, roi_dims = None):

        self.save_video = save_video
        self.show_video = show_video
        self.history_len = history_len

        # Make sure smart_ids are in ascending order
        self.smart_ids = deepcopy(smart_ids)
        self.smart_ids.sort()
        # save frame dimensions and fps as attributes
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

        # initialize camera
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


    def init_camera(self, video_source):
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

    def init_tracking(self):
        '''Initializes April tag detector and creates tracking objects.
        Additionally gets initial position of objects and sets time for t0'''
        # April Tag Detector Object
        self.detector = apriltag("tagStandard41h12")

        # initialize tracking objects
        self.tracking_objects = [Detection(smart_id, self.frame_height, history_length=self.history_len) for smart_id in self.smart_ids]

        # set t0 for tracking data
        self.t0 = time.time()

        for obj in self.tracking_objects:
            det = None
            t_start = time.time()
            while not obj.is_detected(0,det):
                # capture frame and region of interest, specified by crop region
                [self.frame,self.roi] = self.capture_frame()
                # detect april tags in frame
                detections = self.detect_frame(self.roi)
                ids_detected = [x['id'] for x in detections]
                if obj.id not in ids_detected:
                    det = None
                else:
                    det = detections[ids_detected.index(obj.id)]
                    print('Smarticle {} detected in frame'.format(obj.id))
                if (time.time()-t_start)>5:
                    print('Smarticle {}  could not be found in frame'.format(obj.id))
                    break

    def capture_frame(self):
        [x, y, w, h] = self.roi_dims
        ret, frame = self.cap.read()
        roi = frame[y:y+h, x:x+w]
        return [frame,roi]

    def detect_frame(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(gray)

    def save_detections(self):
        # x and y offset of crop region
        offset = self.roi_dims[0:2]
        t= time.time()-self.t0
        ids_detected = [x['id']for x in self.detections]
        for obj in self.tracking_objects:
            if obj.id not in ids_detected:
                obj.add_timestep(t, offset = offset)
            else:
                obj.add_timestep(t, det = self.detections[ids_detected.index(obj.id)], offset = offset)
                if self.show_video is True:
                    cv2.line(self.frame, (int(obj.x[0]),int(obj.frame_height-obj.x[1])),\
                    (int(obj.x[0]+25*np.cos(-obj.x[2])), int(obj.frame_height-obj.x[1]+25*np.sin(-obj.x[2]))),\
                    (0,255,0),5)


    def step(self):
        [self.frame, self.roi] = self.capture_frame()
        if self.save_video is True:
            self.out.write(self.frame)

        self.detections = self.detect_frame(self.roi)
        self.save_detections()

        if self.show_video is True:
            cv2.imshow('frame', self.frame)

    def close(self):
        self.cap.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()

    def save_data(self,path, Ar, Rr, local_copy=False):
        '''
        Saves data to given file path as .csv file in following format
        (time index, len([s,a,r,s']))
        '''
        data = np.hstack([np.array(obj.history) for obj in self.tracking_objects])
        S = data[:-1]
        A = np.array(Ar)
        R = np.array(Rr).reshape(len(Rr),1)
        S_prime = data[1:]
        data_out = np.hstack([S,A,R,S_prime])
        np.savetxt(path,data_out,delimiter=',')

        if local_copy:
            return S,A,R,S_prime

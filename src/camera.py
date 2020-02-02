# camera.py
# OpenCV Camera management (OpenCV--versionTODO)
# Created Feb 1, 2020
# Alex Samland (alexsamland@u.northwestern.edu)

class ClassName(object):


    def __init__(self, video_source=0, save_video=None, frame_width, frame_height,\
        fps, autofocus=0 ,manual_focus=0, brightness=30, contrast=100):
        '''Initializes camera with specified settings as tuned tracking settings
        (ie. turns off autofocus, sets brightness and contrast)'''
        # Camera Settings
        self.video_source = video_source
        self.autofocus = autofocus
        self.focus_level = focus_level
        self.brightness = brightness
        self.contrast = contrast
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap = cv2.VideoCapture(self.video_source) # sets input source for video capture
        self.cap.set(6, self.fourcc) # setting MJPG codec
        self.cap.set(3, frame_width) # Width
        self.cap.set(4, frame_height) # Height
        self.frame_width = self.cap.get(3)
        self.frame_height = self.cap.get(4)
        self.cap.set(5,self.fps) # fps
        self.fps = self.cap.get(5)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, self.autofocus) # disable autofocus
        if self.focus_level is not None:
            self.cap.set(28,self.focus_level) # set focus level
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS,self.brightness) # low brightness
        self.cap.set(cv2.CAP_PROP_CONTRAST,self.contrast) # high contrast
        if self.save_video is not None:
            # Save video to file
            self.out = cv2.VideoWriter(self.save_video,self.fourcc, self.fps, (self.frame_width,self.frame_height))
        else:
            self.out = None

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
        self.ret, self.frame = self.cap.read()
        # save cropped frame
        self.roi = frame[y:y+h, x:x+w]

        return [self.ret, self.frame, self.roi]


    def close(self):
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

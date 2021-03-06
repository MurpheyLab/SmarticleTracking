# camera.py
# OpenCV Camera management (OpenCV--versionTODO)
# Created Feb 1, 2020
# Alex Samland (alexsamland@u.northwestern.edu)

import cv2
import numpy as np

class Camera(object):


    def __init__(self, frame_width, frame_height, fps, video_source=0, save_video=None,\
        show_video=True, roi_dims=None, autofocus=0 ,focus_level=0, brightness=30, contrast=100):
        '''Initializes camera with specified settings as tuned tracking settings
        (ie. turns off autofocus, sets brightness and contrast)

        ## Arguments
        ---

        | Argument     | Type            | Description                                                                             | Default Value  |
        | :------      | :--             | :---------                                                                              |:-----------    |
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
        # Camera Settings
        self.video_source = video_source
        self.save_video = save_video
        self.show_video = show_video
        self.autofocus = autofocus
        self.focus_level = focus_level
        self.brightness = brightness
        self.contrast = contrast
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap = cv2.VideoCapture(self.video_source) # sets input source for video capture
        self.cap.set(6, self.fourcc) # setting MJPG codec
        self.cap.set(3, frame_width) # Width
        self.cap.set(4, frame_height) # Height
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))

        if roi_dims is None:
            self.roi_dims = [0,0,self.frame_width,self.frame_height]
        else:
            assert len(roi_dims) is 4, 'roi_dims is 4 element list of form: [x, y, w, h]'
            self.roi_dims = roi_dims

        self.cap.set(5,fps) # fps
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

    def set_roi_dims(self, center, h, w):
        '''
        ## Description
        ---
        Sets `roi_dims` or region of interest dimension that define a cropped area of the entire captured frame.

        ## Arguments
        ---

        | Argument     | Type            | Description                                                        | Default Value  |
        | :------      | :--             | :---------                                                         | :-----------   |
        | center       | `int`           | (x,y) center of roi (usually corresponds to ring center)           | N/A            |
        | h            | `int`           | height in pixels of roi                                            | N/A            |
        | w            | `int`           | width in pixels of roi                                             | N/A            |
        |<img width=300/>|<img width=300/>|<img width=900/>|<img width=250/>|

        ## Returns
        ---
        None

        '''

        x = int(center[0]-0.5*w)
        np.clip(x,0,self.frame_width)
        y =int(center[1]-0.5*h)
        np.clip(y,0,self.frame_height)

        self.roi_dims=[x,y,int(h), int(w)]

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
        self.roi = self.frame[y:y+h, x:x+w]

        return [self.ret, self.frame, self.roi]

    def write_frame(self, frame=None):
        '''
        ## Description
        ---
        Writes video frame to file specified in class constructor if `self.save_video' is `True`

        ## Arguments
        ---

        | Argument     | Type            | Description                                                        | Default Value  |
        | :------      | :--             | :---------                                                         | :-----------   |
        | frame        | `np.array`      | *Optional:* 3D `np.array` of RGB pixel values. If not provided, `self.frame` is used.          | `None`            |
        |<img width=300/>|<img width=300/>|<img width=900/>|<img width=250/>|

        ## Returns
        ---
        None

        '''
        if frame is None:
            frame = self.frame
        if self.save_video is not None:
            self.out.write(frame)

    def show_frame(self, frame=None):
        '''
        ## Description
        ---
        Displays video frame if `self.show_video' is `True`

        ## Arguments
        ---

        | Argument     | Type            | Description                                                        | Default Value  |
        | :------      | :--             | :---------                                                         | :-----------   |
        | frame        | `np.array`      | *Optional:* 3D `np.array` of RGB pixel values. If not provided, `self.frame` is used.          | `None`            |
        |<img width=300/>|<img width=300/>|<img width=900/>|<img width=250/>|

        ## Returns
        ---
        None

        '''
        if frame is None:
            frame = self.frame
        if self.show_video is True:
            cv2.imshow('frame', frame)




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

# Define a class for working on videos

import cv2
class VideoReader(object):
    '''
    Class docstring for VideoReader():
    Provides a generator for video frames. Returns a numpy array in BGR format. 
    '''
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV parses an integer to read a webcam. Supplying '0' will use webcam. 
            self.file_name = int(file_name)
        except ValueError:
            pass

        self.read = cv2.VideoCapture(self.file_name)
        self.shapes = []
        while(self.read.isOpened()):
            ret, frame = self.read.read()
            if ret == True:
                self.shapes.append(frame.shape) # numpy style shape (H x W x C). 
            else:
                break
        self.read.release()

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

    def shape(self):        
        if not all(self.x == self.shapes[0] for self.x in self.shapes):
            raise IOError('Uneven frame sizes in {}'.format(self.file_name))
        
        return tuple(list(map(max, zip(*self.shapes))))
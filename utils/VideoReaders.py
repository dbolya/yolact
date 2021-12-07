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

    def properties(self):
        self.w, self.h, self.count = self.read.get(cv2.CAP_PROP_FRAME_WIDTH), self.read.get(cv2.CAP_PROP_FRAME_HEIGHT), self.read.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(self.count), (int(self.h), int(self.w))
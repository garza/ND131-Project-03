'''
Simple stream reader wrapped in a while loop with yield to abstract away reading our input stream
'''
import cv2

class FrameFeed:
    def __init__(self, input_type, input_file=None):
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
    
    def load_data(self):
        if self.input_type=='video':
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
        else:
            self.cap=cv2.imread(self.input_file)

    def next_frame(self):
        '''
        As long as we got output from our stream, return the next frame
        Use yield to wait for the next call to next_frame
        '''
        while True:
            for _ in range(10):
                ret, frame=self.cap.read()
            yield ret, frame

    def close(self):
        if not self.input_type=='image':
            self.cap.release()

from threading import Thread
import sys
import cv2
from utils.utils import preprocess_video
if sys.version_info>=(3,0):
    from queue import Queue

else:
    from Queue import Queue
force_input_size = None
compound_coef = 0
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

class FileVideoStream:
    def __init__(self,path,queueSize=15):
        self.stream=cv2.VideoCapture(path)
        self.stopped=False
       
        self.Q=Queue(maxsize=queueSize)
        self.P=Queue(maxsize=queueSize)
        self.R=Queue(maxsize=queueSize)
    def start(self):
        t=Thread(target=self.update,args=())
        t.daemon=True
        t.start()
        return self
    def update(self):
        while True:
            if self.stopped:
                return
            if  self.Q.qsize()<80:
                grabbed,frame=self.stream.read()
                
                if grabbed:
                    #frame=cv2.resize(frame,(300,300),interpolation=cv2.INTER_LINEAR)
                    ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)
                    self.Q.put(framed_imgs)
                    self.P.put(ori_imgs)
                    self.R.put(framed_metas)
                else:
                    self.stream.set(cv2.CAP_PROP_POS_FRAMES,0)
    
    def read(self):
        return self.Q.get(),self.P.get(),self.R.get()
    def more(self):
        return self.Q.qsize()>0
    def stop(self):
        self.stopped=True
    

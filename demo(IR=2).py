import time
import torch
import cv2
from queue import Queue
from utils.ReadSource import FileVideoStream
from threading import Thread
import numpy as np
from torch.backends import cudnn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video
def post(postprocess,x,model,regressBoxes,clipBoxes,threshold,iou_threshold,framed_metas,invert_affine,display,ori,res):
        with torch.no_grad():
            features, regression, classification, anchors = model(x)
            
            for i in range(0,2):
                lo=[]
                #print(i)
                out = postprocess(x[i],
                                anchors, regression[i], classification[i],
                                regressBoxes, clipBoxes,
                                threshold, iou_threshold)
                #print(framed_metas[i])
                lo.append(framed_metas[i])
                out = invert_affine(lo, out)
                img_show = display(out, ori[i])
                res.put(img_show)
# Video's path
video_src = '1.mp4'  # set int to use webcam, set str to read from a video file
video_src1 = '2.mp4'
video_src2 = '3.mp4'
video_src3 = '4.mp4'
compound_coef = 0
force_input_size = None  # set None to use default size

threshold = 0.25
iou_threshold = 0.25

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

# load model
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
model.load_state_dict(torch.load(f'weights/efficientdet-d0.pth'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

# function for display
def display(preds, imgs):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            return imgs[i]

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        
        return imgs[i]
# Box
regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

# Video capture
fvs1 = FileVideoStream(video_src).start()
time.sleep(1.0)
fvs2 = FileVideoStream(video_src1).start()
time.sleep(1.0)
fvs3 = FileVideoStream(video_src2).start()
time.sleep(1.0)
fvs4 = FileVideoStream(video_src3).start()
time.sleep(1.0)
frame_counter = 0
frame_counter1 = 0
frame_counter2 = 0
frame_counter3 = 0
op=True
res1 = Queue()
res2 = Queue()
#res3 = Queue()
#res4 = Queue()
while True:
    
        stime=time.time()
        framed_imgs,ori_imgs,framed_metas = fvs1.read()
        framed_imgs1,ori_imgs1,framed_metas1 = fvs2.read()
        framed_imgs2,ori_imgs2,framed_metas2 = fvs3.read()
        framed_imgs3,ori_imgs3,framed_metas3 = fvs4.read()
        time.sleep(0.03)
        ############################STACK 1 END~~
        ori1=[]
        ori2=[]
        framed_imgs.append(framed_imgs1[0])
        framed_metas.append(framed_metas1[0])
        ori1.append(ori_imgs)
        ori1.append(ori_imgs1)   
        framed_imgs2.append(framed_imgs3[0])
        framed_metas2.append(framed_metas3[0])
        ori2.append(ori_imgs2)
        ori2.append(ori_imgs3) 
        x1 = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        x1 = x1.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
        x2 = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs2], 0)
        x2 = x2.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
        # model predict
        #print(framed_metas)
        #print(framed_metas)
        t1 = Thread(target = post, args =(postprocess,x1,model,regressBoxes,clipBoxes,threshold,iou_threshold,framed_metas,invert_affine,display,ori1,res1, ))
        t2 = Thread(target = post, args =(postprocess,x2,model,regressBoxes,clipBoxes,threshold,iou_threshold,framed_metas2,invert_affine,display,ori2,res2, ))
        t1.start()
        t2.start()
        t1.join()
        img1=res1.get()
        img2=res1.get()
        t2.join()
        img3=res2.get()
        img4=res2.get()
        
        cv2.imshow("img1",img1)
        cv2.imshow("img2",img2)
        cv2.imshow("img3",img3)
        cv2.imshow("img4",img4)
        res1.queue.clear()
        res2.queue.clear() 
        print('FPS {:.1f}'.format(1/ (time.time() - stime)))
    
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

cv2.destroyAllWindows()
#CLEAN UP
fvs1.stop()
fvs2.stop()
fvs3.stop()
fvs4.stop()






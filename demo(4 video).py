import time
import torch
import cv2
from utils.ReadSource import FileVideoStream
from queue import Queue
from threading import Thread
import numpy as np
from torch.backends import cudnn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video

#POSTPROC & DISPLAY
def post(postprocess,x,anchors,regression,classification,regressBoxes,clipBoxes,threshold,iou_threshold,lo,invert_affine,display,ori,res):
                out = postprocess(x,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                threshold, iou_threshold)            
                out = invert_affine(lo, out)
                img_show = display(out, ori)
                img_show=cv2.resize(img_show,(960,540),interpolation=cv2.INTER_LINEAR)
                res.put(img_show)
# Video's path
video_src = '1.mp4'  
video_src1 = '2.mp4'
video_src2 = '3.mp4'
video_src3 = '4.mp4'
compound_coef = 0
force_input_size = None  

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


input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

#LOAD MODEL
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
model.load_state_dict(torch.load(f'weights/efficientdet-d0.pth'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

#BB&PT
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
#BOX
regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

#START VIDEO READ THREADS
fvs1 = FileVideoStream(video_src).start()
time.sleep(1.0)
fvs2 = FileVideoStream(video_src1).start()
time.sleep(1.0)
fvs3 = FileVideoStream(video_src2).start()
time.sleep(1.0)
fvs4 = FileVideoStream(video_src3).start()
time.sleep(1.0)
res1 = Queue()
res2 = Queue()
res3 = Queue()
res4 = Queue()
while True:
        #GET FRAME
        stime=time.time()
        framed_imgs,ori_imgs,framed_metas = fvs1.read()
        framed_imgs1,ori_imgs1,framed_metas1 = fvs2.read()
        framed_imgs2,ori_imgs2,framed_metas2 = fvs3.read()
        framed_imgs3,ori_imgs3,framed_metas3 = fvs4.read()
        #time.sleep(0.03)
        
        #BATCHING
        framed_imgs.append(framed_imgs1[0])
        framed_imgs.append(framed_imgs2[0])
        framed_imgs.append(framed_imgs3[0])  
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)    
        #PREDICTING
        with torch.no_grad():     
            features, regression, classification, anchors = model(x)
            t1 = Thread(target = post, args =(postprocess,x[0],anchors,regression[0],classification[0],regressBoxes,clipBoxes,threshold,iou_threshold,framed_metas,invert_affine,display,ori_imgs,res1, ))
            t2 = Thread(target = post, args =(postprocess,x[1],anchors,regression[1],classification[1],regressBoxes,clipBoxes,threshold,iou_threshold,framed_metas1,invert_affine,display,ori_imgs1,res2, ))
            t3 = Thread(target = post, args =(postprocess,x[2],anchors,regression[2],classification[2],regressBoxes,clipBoxes,threshold,iou_threshold,framed_metas2,invert_affine,display,ori_imgs2,res3, ))
            t4 = Thread(target = post, args =(postprocess,x[3],anchors,regression[3],classification[3],regressBoxes,clipBoxes,threshold,iou_threshold,framed_metas3,invert_affine,display,ori_imgs3,res4, ))
            t1.start()
            t2.start()
            t3.start()
            t4.start()
            t1.join()
            img1=res1.get()
            t2.join()
            img2=res2.get()
            numpy_horizontal1 = np.hstack((img1,img2))
            t3.join()
            img3=res3.get()
            t4.join()
            img4=res4.get()             
            numpy_horizontal2 = np.hstack((img3,img4))
            numpy_verical=np.vstack((numpy_horizontal1,numpy_horizontal2))
            cv2.imshow("RESULT",numpy_verical)
            #cv2.imshow("img1",img1)
            #cv2.imshow("img2",img2)
            #cv2.imshow("img3",img3)
            #cv2.imshow("img4",img4)
        print('FPS {:.1f}'.format(1/ (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break


cv2.destroyAllWindows()
#CLEAN UP
fvs1.stop()
fvs2.stop()
fvs3.stop()
fvs4.stop()





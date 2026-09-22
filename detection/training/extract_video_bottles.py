"""Detect bottles, track with camera motion, and save best original-resolution crops."""
import argparse,json,math,os
from pathlib import Path
os.environ.setdefault('YOLO_CONFIG_DIR','/tmp/wine_yolo_config')
import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


def motion(previous,current):
    if previous is None: return np.eye(3)
    pts=cv2.goodFeaturesToTrack(previous,900,.01,8)
    if pts is None or len(pts)<12: return np.eye(3)
    nxt,status,_=cv2.calcOpticalFlowPyrLK(previous,current,pts,None,winSize=(31,31),maxLevel=4)
    back,st2,_=cv2.calcOpticalFlowPyrLK(current,previous,nxt,None,winSize=(31,31),maxLevel=4)
    good=(status.ravel()>0)&(st2.ravel()>0)&(np.linalg.norm(pts-back,axis=2).ravel()<2)
    if good.sum()<12:return np.eye(3)
    matrix,inliers=cv2.estimateAffinePartial2D(pts[good],nxt[good],method=cv2.RANSAC,ransacReprojThreshold=3)
    if matrix is None or inliers.sum()<10:return np.eye(3)
    return np.vstack([matrix,[0,0,1]])


def warp(box,matrix):
    x,y,X,Y=box
    p=np.array([[x,y,1],[X,y,1],[X,Y,1],[x,Y,1]])@matrix.T
    return np.r_[p[:,:2].min(0),p[:,:2].max(0)]


def overlap(a,b):
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return inter/max(1,np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter)


def appearance(crop):
    small=cv2.resize(crop,(24,48));hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1],None,[16,8],[0,180,0,256]);return cv2.normalize(hist,hist).flatten()


def quality(crop,box,shape,conf):
    h,w=crop.shape[:2]
    resized=cv2.resize(crop,(128,384))
    label=cv2.cvtColor(resized[150:345,12:116],cv2.COLOR_BGR2GRAY)
    sharp=float(cv2.Laplacian(label,cv2.CV_32F).var())
    margin=min(box[0],box[1],shape[1]-box[2],shape[0]-box[3])
    edge=.22 if margin<5 else (.65 if margin<18 else 1.)
    score=math.log1p(sharp)*min(1.5,math.sqrt(w*h/180000))*edge*float(conf)
    return score,sharp,margin


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='detection/runs/grain_yolo26l-2/weights/best.pt')
    p.add_argument('--device',default='cpu');p.add_argument('--fps',type=float,default=5);p.add_argument('--imgsz',type=int,default=960)
    args=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(4)
    args.output.mkdir(parents=True,exist_ok=True);crops=args.output/'crops';crops.mkdir(exist_ok=True)
    if (args.output/'manifest.json').exists():raise SystemExit('Output already contains a finished run; use a new directory')
    cap=cv2.VideoCapture(str(args.video));fps=cap.get(cv2.CAP_PROP_FPS);step=max(1,round(fps/args.fps))
    model=YOLO(args.model);tracks=[];previous=None;frame_no=-1;observations=0;sampled=0
    while True:
        ok=cap.grab();frame_no+=1
        if not ok:break
        if frame_no%step:continue
        ok,frame=cap.retrieve()
        if not ok:continue
        sampled+=1;timestamp=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        gray=cv2.cvtColor(cv2.resize(frame,(0,0),fx=.5,fy=.5),cv2.COLOR_BGR2GRAY)
        transform=motion(previous,gray);transform[:2,2]*=2;previous=gray
        active=[t for t in tracks if timestamp-t['last_time']<2.5]
        for t in active:t['pred']=warp(t['pred'],transform)
        result=model.predict(frame,imgsz=args.imgsz,conf=.30,iou=.45,device=args.device,verbose=False)[0]
        detections=[]
        for box,conf in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy()):
            x,y,X,Y=np.rint(box).astype(int);x=max(0,x);y=max(0,y);X=min(frame.shape[1],X);Y=min(frame.shape[0],Y)
            if X-x<35 or Y-y<90:continue
            crop=frame[y:Y,x:X];detections.append((np.array([x,y,X,Y],float),float(conf),crop,appearance(crop)))
        costs=np.ones((len(active),len(detections)))*100
        for i,t in enumerate(active):
            for j,(box,conf,crop,hist) in enumerate(detections):
                iou=overlap(t['pred'],box)
                dist=np.linalg.norm(((t['pred'][:2]+t['pred'][2:])-(box[:2]+box[2:]))/2/np.maximum(box[2:]-box[:2],1))
                similarity=float(np.dot(t['hist'],hist))
                if iou>.15 and dist<.8 and similarity>.35:costs[i,j]=1-iou+.25*(1-similarity)
        matches={}
        if costs.size:
            rows,cols=linear_sum_assignment(costs)
            matches={j:active[i] for i,j in zip(rows,cols) if costs[i,j]<1.1}
        for j,(box,conf,crop,hist) in enumerate(detections):
            t=matches.get(j)
            if t is None:
                t={'id':len(tracks)+1,'first_time':timestamp,'hits':0,'best_score':-1,'history':[]};tracks.append(t)
            t.update(pred=box,hist=hist,last_time=timestamp);t['hits']+=1
            score,sharp,margin=quality(crop,box,frame.shape,conf)
            t['history'].append({'frame':frame_no,'time':round(timestamp,3),'box':box.tolist(),'confidence':conf,'score':score})
            if score>t['best_score']:
                t.update(best_score=score,best_frame=frame_no,best_time=timestamp,best_box=box.tolist(),sharpness=sharp,margin=float(margin),confidence=conf)
                cv2.imwrite(str(crops/f"bottle_{t['id']:04d}.jpg"),crop,[cv2.IMWRITE_JPEG_QUALITY,97])
            observations+=1
        if sampled%20==0:print(f'{timestamp:.1f}s: {sampled} frames, {len(tracks)} tracks, {observations} detections',flush=True)
    cap.release()
    accepted=[];rejected=[]
    uncertain=args.output/'uncertain';uncertain.mkdir(exist_ok=True)
    for t in tracks:
        t.pop('pred',None);t.pop('hist',None)
        # Retain short/edge-only tracks separately instead of silently losing bottles.
        valid=t['hits']>=3 and t['margin']>=5
        folder=crops if valid else uncertain
        name=f"bottle_{t['id']:04d}.jpg"
        if not valid:(crops/name).rename(folder/name)
        t['crop']=str(folder.relative_to(args.output)/name)
        (accepted if valid else rejected).append(t)
    manifest={'video':str(args.video.resolve()),'model':args.model,'source_fps':fps,'sample_step':step,'sampled_frames':sampled,'detections':observations,'tracks':accepted,'uncertain_tracks':rejected,'notes':'One crop per motion-compensated track; repeated visits may produce duplicate tracks. Hidden bottles cannot be recovered. Best among sampled frames, not every source frame.'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # Self-contained local gallery referencing full-resolution JPEGs.
    html=['<!doctype html><meta charset="utf-8"><title>Bottle crops</title><style>body{background:#eee;font:15px sans-serif}section{display:flex;flex-wrap:wrap}figure{margin:8px;background:white;padding:8px}img{width:135px;height:300px;object-fit:contain}</style>']
    for title,items in [('Bottle tracks',accepted),('Uncertain: short or clipped tracks',rejected)]:
        html.append(f'<h1>{title}: {len(items)}</h1><section>')
        for t in items:html.append(f'<figure><a href="{t["crop"]}"><img loading="lazy" src="{t["crop"]}"></a><figcaption>#{t["id"]} · {t["best_time"]:.2f}s<br>{t["hits"]} observations</figcaption></figure>')
        html.append('</section>')
    (args.output/'index.html').write_text('\n'.join(html))
    print(f'DONE: {len(accepted)} crops, {len(rejected)} uncertain; {args.output}',flush=True)

if __name__=='__main__':main()

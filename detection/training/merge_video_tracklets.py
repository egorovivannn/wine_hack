"""Conservatively join non-overlapping nearby track fragments using local features."""
import json,sys
from pathlib import Path
import cv2
import numpy as np
root=Path(sys.argv[1]);path=root/'manifest.json';data=json.loads(path.read_text())
tracks=data['tracks']+data['uncertain_tracks'];sift=cv2.SIFT_create(nfeatures=1600);features={}
for t in tracks:
 im=cv2.imread(str(root/t['crop']),0);scale=min(1,600/im.shape[0]);im=cv2.resize(im,None,fx=scale,fy=scale)
 k,d=sift.detectAndCompute(im,None);features[t['id']]=(k,d,im.shape)
parent={t['id']:t['id'] for t in tracks};members={t['id']:[t] for t in tracks}
def rep(i):
 while parent[i]!=i:i=parent[i]
 return i
def disjoint(a,b):
 return a['last_time']<b['first_time'] or b['last_time']<a['first_time']
candidates=[];matcher=cv2.BFMatcher()
for i,a in enumerate(tracks):
 for b in tracks[i+1:]:
  if not disjoint(a,b):continue
  gap=max(a['first_time'],b['first_time'])-min(a['last_time'],b['last_time'])
  if gap>1.2:continue
  ka,da,sa=features[a['id']];kb,db,sb=features[b['id']]
  if da is None or db is None or min(len(da),len(db))<20:continue
  good=[m for pair in matcher.knnMatch(da,db,k=2) if len(pair)==2 for m,n in [pair] if m.distance<.65*n.distance]
  if len(good)<20:continue
  pa=np.float32([ka[m.queryIdx].pt for m in good]);pb=np.float32([kb[m.trainIdx].pt for m in good])
  h,mask=cv2.findHomography(pa,pb,cv2.RANSAC,3)
  if h is None:continue
  count=int(mask.sum());ratio=count/len(good)
  pts=pa[mask.ravel()>0];coverage=np.ptp(pts,axis=0)/[sa[1],sa[0]]
  if count>=18 and ratio>=.65 and min(coverage)>.3:candidates.append((count*ratio,a,b))
merges=[]
for confidence,a,b in sorted(candidates,key=lambda x:-x[0]):
 ra,rb=rep(a['id']),rep(b['id'])
 if ra==rb:continue
 if not all(disjoint(x,y) for x in members[ra] for y in members[rb]):continue
 parent[rb]=ra;members[ra]+=members.pop(rb);merges.append([a['id'],b['id'],confidence])
(root/'merged_duplicates').mkdir(exist_ok=True)
accepted=[];uncertain=[]
for group in members.values():
 best=max(group,key=lambda t:t['best_score']);best['merged_track_ids']=[t['id'] for t in group]
 for t in group:
  if t is not best:
   src=root/t['crop'];src.rename(root/'merged_duplicates'/src.name)
 best['first_time']=min(t['first_time'] for t in group);best['last_time']=max(t['last_time'] for t in group)
 history=sorted([h for t in group for h in t['history']],key=lambda h:h['frame']);best['history']=history;best['hits']=len(history)
 valid=best['hits']>=3 and best['margin']>=5
 folder='crops' if valid else 'uncertain';src=root/best['crop'];dst=root/folder/src.name
 if src!=dst:src.rename(dst)
 best['crop']=str(dst.relative_to(root));(accepted if valid else uncertain).append(best)
data['tracks']=sorted(accepted,key=lambda t:t['id']);data['uncertain_tracks']=sorted(uncertain,key=lambda t:t['id']);data['tracklet_merges']=merges
(root/'manifest_before_merge.json').write_text(path.read_text());path.write_text(json.dumps(data,indent=2))
html=['<!doctype html><meta charset="utf-8"><title>Wine bottle crops</title><style>body{font:16px sans-serif;background:#eee}section{display:flex;flex-wrap:wrap}figure{margin:8px;padding:8px;background:white}img{width:140px;height:320px;object-fit:contain}</style>']
for title,items in [('Основные кропы',data['tracks']),('Неполные / короткие треки',data['uncertain_tracks'])]:
 html.append(f'<h1>{title}: {len(items)}</h1><section>')
 for t in items:html.append(f'<figure><a href="{t["crop"]}"><img loading="lazy" src="{t["crop"]}"></a><figcaption>#{t["id"]} · {t["best_time"]:.2f}s<br>{t["hits"]} кадров</figcaption></figure>')
 html.append('</section>')
(root/'index.html').write_text('\n'.join(html))
print('Merged',len(merges),'fragments;',len(accepted),'main;',len(uncertain),'uncertain')

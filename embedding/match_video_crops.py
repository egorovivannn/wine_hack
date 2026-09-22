"""Match video crops to unique df_2 catalog images using the full trained embedder."""
import argparse,csv,hashlib,html,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from embedding.inference import WineEmbedder
from bottle_embeddings import read_image,crop_bottle


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--csv',type=Path,required=True)
 p.add_argument('--queries',type=Path,required=True);p.add_argument('--images',type=Path,default=Path('data/competition_imgs'))
 p.add_argument('--boxes',type=Path,default=Path('df_2_bboxes.json'));p.add_argument('--output',type=Path,required=True)
 p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=16)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
 for folder in ['gallery','queries','per_query']:(a.output/folder).mkdir()
 started=time.time();frame=pd.read_csv(a.csv).fillna('');boxes=json.loads(a.boxes.read_text());records=[];excluded=[];pending=[];vectors=[]
 model=WineEmbedder(a.checkpoint,device=a.device)
 def flush():
  if pending:vectors.append(model.embed(pending,batch_size=a.batch_size));pending.clear()
 for fname,group in frame.groupby('fname',sort=False):
  path=a.images/fname
  try:im=read_image(path)
  except Exception as e:excluded.append({'fname':fname,'reason':str(e)});continue
  entries=boxes.get(fname) or [{'xyxy':[0,0,*im.size],'fallback_full_image':True}]
  for bi,b in enumerate(entries):
   crop,xyxy=crop_bottle(im,b['xyxy']);idx=len(records)
   thumb=crop.copy();thumb.thumbnail((220,500));thumb.save(a.output/'gallery'/f'{idx:05d}.jpg',quality=90)
   records.append({'fname':fname,'box_index':bi,'bbox':xyxy,'fallback_full_image':b.get('fallback_full_image',False),'image_path':str(path.resolve()),'preview':f'gallery/{idx:05d}.jpg','csv_rows':group.index.tolist(),'metadata':group.iloc[0].to_dict()})
   pending.append(crop)
   if len(pending)==a.batch_size:flush()
  if len(records)%100<3:print('Gallery',len(records),flush=True)
 flush();g=np.concatenate(vectors).astype(np.float32);g/=np.linalg.norm(g,axis=1,keepdims=True)
 paths=sorted(x for x in a.queries.iterdir() if x.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
 q=model.embed(paths,batch_size=a.batch_size).astype(np.float32);q/=np.linalg.norm(q,axis=1,keepdims=True)
 assert np.isfinite(g).all() and np.isfinite(q).all()
 np.save(a.output/'gallery_embeddings.npy',g);np.save(a.output/'query_embeddings.npy',q)
 (a.output/'gallery_records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
 similarities=q@g.T;results=[];flat=[]
 page=['<!doctype html><meta charset="utf-8"><title>Wine top 5</title><style>body{font:14px sans-serif;background:#eee;margin:24px}section{display:flex;background:white;margin:20px 0;padding:12px;gap:14px}figure{margin:0;width:180px}img{width:170px;height:340px;object-fit:contain}figcaption{overflow-wrap:anywhere}.query{border-right:3px solid #bbb;padding-right:14px}</style><h1>Кроп → топ‑5 изображений df_2</h1><p>Cosine similarity — сходство эмбеддингов, не вероятность распознавания. Пять разных файлов каталога; названия вин могут повторяться.</p>']
 for qi,path in enumerate(paths):
  im=Image.open(path).convert('RGB');im.thumbnail((220,500));im.save(a.output/'queries'/f'{path.stem}.jpg',quality=90)
  seen=set();matches=[]
  for gi in np.argsort(-similarities[qi],kind='stable'):
   r=records[gi]
   if r['fname'] in seen:continue
   seen.add(r['fname']);rank=len(matches)+1;score=float(similarities[qi,gi]);match={'rank':rank,'score':score,**r};matches.append(match)
   flat.append({'query':path.name,'query_path':str(path.resolve()),'rank':rank,'cosine_similarity':score,'fname':r['fname'],'vine_name':r['metadata']['vine_name'],'slug':r['metadata']['slug'],'image_path':r['image_path'],'box_index':r['box_index'],'fallback_full_image':r['fallback_full_image'],'csv_rows':json.dumps(r['csv_rows'])})
   if len(matches)==5:break
  item={'query':path.name,'query_path':str(path.resolve()),'matches':matches};results.append(item)
  (a.output/'per_query'/f'{path.stem}.json').write_text(json.dumps(item,ensure_ascii=False,indent=2))
  page.append(f'<section><figure class="query"><a href="{html.escape(path.resolve().as_uri())}"><img loading="lazy" src="queries/{path.stem}.jpg"></a><figcaption>{html.escape(path.name)}</figcaption></figure>')
  for m in matches:page.append(f'<figure><a href="{html.escape(Path(m["image_path"]).as_uri())}"><img loading="lazy" src="{m["preview"]}"></a><figcaption>#{m["rank"]} · {m["score"]:.4f}<br>{html.escape(m["metadata"]["vine_name"])}<br>{html.escape(m["fname"])}</figcaption></figure>')
  page.append('</section>')
 pd.DataFrame(flat).to_csv(a.output/'top5.csv',index=False)
 (a.output/'top5.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));(a.output/'index.html').write_text('\n'.join(page))
 meta={'checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':hashlib.file_digest(a.checkpoint.open('rb'),'sha256').hexdigest(),'csv':str(a.csv.resolve()),'csv_sha256':hashlib.sha256(a.csv.read_bytes()).hexdigest(),'bbox_sha256':hashlib.sha256(a.boxes.read_bytes()).hexdigest(),'dimension':int(g.shape[1]),'queries':len(paths),'catalog_unique_images':len(set(r['fname'] for r in records)),'gallery_crops':len(records),'full_image_fallbacks':sum(r['fallback_full_image'] for r in records),'excluded':excluded,'ranking':'exact float32 cosine; max similarity across boxes per unique fname; repeated CSV rows grouped','preprocessing':'checkpoint-configured BottleTransform; full backbone + projection head','elapsed_seconds':time.time()-started}
 (a.output/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)
 assert len(results)==len(paths) and all(len(r['matches'])==5 for r in results)
 assert all(len(set(m['fname'] for m in r['matches']))==5 for r in results)
 print('DONE',a.output,flush=True)

if __name__=='__main__':main()

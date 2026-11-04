"""Link existing ranked thumbnails to original sampled windows; no model inference.

Uses exact pixel comparisons after the documented 1024-to-220 resize. Outputs
publication-coded assets and coordinates. Original source files remain untouched.
"""
from pathlib import Path
import csv, json, hashlib, shutil
import numpy as np
import h5py
import openslide
from PIL import Image

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

split=read(PROJECT/'scripts/result/ss_experiments/malt_vs_non_malt/uni_v2_hybrid_evidence_mil/cohort_split.csv')
old=json.loads((ROOT.parent/'draft_v1/source_data/hybrid_cases_manifest.json').read_text())
cases=read(ROOT/'supporting_data/interpretation_cases.csv')
out=[];integrity=[]
for num in [1,4,7,10]:
    entry=old[num-1];code=cases[num-1]['study_slide_id'];sid=entry['slide_id']
    idx,row=next((idx,row) for idx,row in enumerate(split) if row['slide_id']==sid)
    folder=Path(entry['panel']).parent
    targets=[]
    for direction,prefix in [('positive','malt_like_'),('negative','nonmalt_like_')]:
        src=folder/f'{sid}_{prefix}01.png';dst=ROOT/'assets'/f'case_{num:02d}_{direction}.png'
        shutil.copy2(src,dst);targets.append(np.asarray(Image.open(src).convert('RGB')))
        integrity.append({'asset':str(dst.relative_to(ROOT)),'source_sha256':sha(src),'asset_sha256':sha(dst),'operation':'byte-identical copy'})
    src=folder/f'{sid}_malt_margin_overlay.png';dst=ROOT/'assets'/f'case_{num:02d}_margin.png'
    shutil.copy2(src,dst);integrity.append({'asset':str(dst.relative_to(ROOT)),'source_sha256':sha(src),'asset_sha256':sha(dst),'operation':'byte-identical copy'})
    with h5py.File(row['feature_path'],'r') as f:
        coords=f['coords'][:];attrs=dict(f['coords'].attrs)
    keep=np.random.default_rng(42+idx).choice(len(coords),512,replace=len(coords)<512)
    selected=coords[keep]
    slide=openslide.OpenSlide(row['slide_path'])
    matches=[[],[]];best=[float('inf'),float('inf')]
    for j,(x,y) in enumerate(selected):
        im=slide.read_region((int(x),int(y)),0,(1024,1024)).convert('RGB').resize((220,220),Image.Resampling.LANCZOS)
        a=np.asarray(im)
        for k,target in enumerate(targets):
            err=float(np.mean(np.abs(a.astype(float)-target)))
            best[k]=min(best[k],err)
            if np.array_equal(a,target):matches[k].append((int(x),int(y),j))
        if all(matches):break
    slide.close()
    assert all(matches),(num,best)
    rec={'publication_case':f'Case {num}','study_slide_id':code,'level0_width':int(attrs['level0_width']),
         'level0_height':int(attrs['level0_height']),'patch_width_level0':1024,
         'patch_width_um':1024*0.2257081593499605,'display_width_pixels':220,'match':'exact RGB pixel equality after documented resize'}
    for k,direction in enumerate(['positive','negative']):
        rec[direction+'_x'],rec[direction+'_y'],rec[direction+'_sample_index']=matches[k][0]
    out.append(rec)
    print(f'Case {num}: both ranked thumbnails matched exactly',flush=True)

native=Image.open(ROOT.parent/'draft_v1/assets/image9.png').convert('RGB')
for name,box in [('malt',(26,96,1306,897)),('ss',(1356,96,2641,897))]:
    dst=ROOT/'assets'/f'illustrative_{name}.png';native.crop(box).save(dst)
    assert np.array_equal(np.asarray(Image.open(dst)),np.asarray(native)[box[1]:box[3],box[0]:box[2]])
    integrity.append({'asset':str(dst.relative_to(ROOT)),'source_sha256':sha(ROOT.parent/'draft_v1/assets/image9.png'),
        'crop_box':box,'asset_sha256':sha(dst),'operation':'unchanged rectangular crop; source RGB pixels retained'})
(ROOT/'source_data/roi_locations.json').write_text(json.dumps(out,indent=2)+'\n')
(ROOT/'source_data/image_integrity_v5.json').write_text(json.dumps({'assets':integrity,'brightness_contrast_gamma':'unchanged','model_inference_runs':0,'tissue_synthesis':'none'},indent=2)+'\n')
print('All ROI locations recovered without model inference.',flush=True)

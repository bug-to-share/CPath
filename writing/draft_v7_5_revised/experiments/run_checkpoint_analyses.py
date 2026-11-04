#!/usr/bin/env python
"""Reproduce frozen predictions and assess sampling variability without training."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, os, platform, sys, time
from datetime import datetime, timezone
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
DRAFT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import ss_experiment_suite as suite

MODELS = ['hybrid_evidence_mil', 'abmil', 'clam_sb', 'rrtmil']
BRANCHES = ['attention', 'mean', 'std', 'high_margin', 'low_margin_plus_region']

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def prepare_manifest():
    mapping = pd.read_csv(DRAFT/'qa/slide_code_mapping_PRIVATE.csv')
    imap = mapping[mapping.scope.eq('internal')].set_index('original_slide_id').study_slide_id.to_dict()
    emap = mapping[mapping.scope.eq('external')].set_index('original_slide_id').study_slide_id.to_dict()
    base = ROOT/'scripts/result/ss_experiments/malt_vs_non_malt'
    internal = pd.read_csv(base/'uni_v2_hybrid_evidence_mil/cohort_split.csv')
    internal = internal[internal.split.eq('test')].reset_index(drop=True)
    all_external = pd.read_csv(ROOT/'scripts/result/external_multicenter_hybrid_all/external_manifest.csv')
    final = pd.read_csv(DRAFT/'supporting_data/external_final_predictions.csv').set_index('study_slide_id')
    records=[]
    for idx, row in internal.iterrows():
        records.append(dict(scope='internal',study_slide_id=imap[row.slide_id],label=int(row.label),original_index=int(idx),
                            original_slide_id=row.slide_id, feature_path=row.feature_path,reference_seed=42+int(idx)))
    for idx, row in all_external.iterrows():
        sid=emap.get(row.slide_id)
        if sid not in final.index: continue
        assert int(row.label)==int(final.loc[sid,'label'])
        records.append(dict(scope='external',study_slide_id=sid,label=int(row.label),original_index=int(idx),
                            original_slide_id=row.slide_id,feature_path=str(ROOT/'scripts/result/external_multicenter_hybrid_all/20x_512px_0px_overlap/features_uni_v2'/f'{row.slide_id}.h5'),reference_seed=42+int(idx)))
    df=pd.DataFrame(records)
    assert df.groupby('scope').size().to_dict()=={'external':168,'internal':49}
    assert df.study_slide_id.nunique()==217
    assert df.feature_path.map(lambda p:Path(p).is_file()).all()
    return df

def pick(n, seed, scope):
    ids=np.random.default_rng(seed).choice(n,512,replace=n<512)
    if scope=='internal' and n>=512: ids=np.sort(ids)
    return ids

def repeat_seed(scope,sid,rep):
    return int.from_bytes(hashlib.sha256(f'20261104|{scope}|{sid}|{rep}'.encode()).digest()[:4],'little')

def diagnostics(model, batch):
    enc=model.encoder
    h=enc.project(batch['features'].to(next(model.parameters()).device))
    a,_=enc.attention(h)
    sim=torch.einsum('bnh,ckh->bnck',F.normalize(h,dim=-1),F.normalize(enc.prototypes,dim=-1))
    banks=sim.max(dim=-1).values
    margin=banks[...,1]-banks[...,0]
    k=min(32,h.size(1))
    hi=margin.topk(k,dim=1).indices;lo=(-margin).topk(k,dim=1).indices
    high=torch.gather(h,1,hi.unsqueeze(-1).expand(-1,-1,h.size(-1))).mean(dim=1)
    low=torch.gather(h,1,lo.unsqueeze(-1).expand(-1,-1,h.size(-1))).mean(dim=1)
    xy=batch['coords'].to(h.device)
    order=torch.argsort(xy[...,1]*1000000+xy[...,0],dim=1)
    ordered=torch.gather(h,1,order.unsqueeze(-1).expand(-1,-1,h.size(-1)))
    tokens=torch.stack([x.mean(dim=1) for x in torch.chunk(ordered,min(enc.regions,h.size(1)),dim=1)],dim=1)
    region,_=enc.region_pool(tokens)
    stats=torch.cat([banks[...,0].topk(k,dim=1).values.mean(dim=1,keepdim=True),banks[...,1].topk(k,dim=1).values.mean(dim=1,keepdim=True),
        margin.mean(dim=1,keepdim=True),margin.std(dim=1,keepdim=True),margin.max(dim=1).values[:,None],margin.min(dim=1).values[:,None],
        (margin>0).float().mean(dim=1,keepdim=True),(margin>margin.mean(dim=1,keepdim=True)).float().mean(dim=1,keepdim=True)],dim=1)
    vec=torch.stack([a,h.mean(dim=1),h.std(dim=1),high,low+region],dim=1)
    gate=torch.softmax(enc.gate(torch.cat([vec.flatten(1),stats],dim=1)),dim=1)
    logit=model.classifier(enc.fusion(torch.cat([(vec*gate[:,:,None]).flatten(1),stats],dim=1))).squeeze(1)
    result={'instrumented_logit':float(logit.item()),'margin_mean':float(margin.mean()),'margin_sd':float(margin.std()),
            'margin_min':float(margin.min()),'margin_max':float(margin.max()),'positive_margin_fraction':float((margin>0).float().mean())}
    for j,name in enumerate(BRANCHES):
        result['gate_'+name]=float(gate[0,j]);result['norm_'+name]=float(vec[0,j].norm())
        result['scaled_norm_'+name]=float((vec*gate[:,:,None])[0,j].norm())
    for bank in range(2):
        counts=torch.bincount(sim[0,:,bank,:].argmax(-1),minlength=8).cpu().tolist()
        for p,count in enumerate(counts):result[f'bank{bank}_prototype{p}_winner_count']=count
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--batch-size',type=int,default=8)
    args=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3', 'Only physical GPU 3 is authorized'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(20261104);np.random.seed(20261104)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    out=Path(__file__).resolve().parent/('pilot' if args.pilot else 'outputs');out.mkdir(exist_ok=True)
    started=time.monotonic();meta={'started_utc':datetime.now(timezone.utc).isoformat(),'current_environment_not_historical':True,
        'python':platform.python_version(),'pytorch':torch.__version__,'cuda_runtime':torch.version.cuda,
        'gpu_name':torch.cuda.get_device_name(0),'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),
        'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'dtype':'float32','batch_size':args.batch_size,
        'packages':{k:importlib.metadata.version(k) for k in ['numpy','pandas','h5py','scikit-learn','scipy']},
        'runner_sha256':sha(__file__),'protocol_sha256':sha(Path(__file__).with_name('PROTOCOL.md')),
        'source_sha256':sha(ROOT/'scripts/ss_experiment_suite.py')}
    manifest=prepare_manifest()
    if args.pilot:manifest=manifest.groupby('scope',sort=False).head(3)
    private=out/'PRIVATE';private.mkdir(exist_ok=True);manifest.to_csv(private/'input_manifest.csv',index=False)
    models={};modelmeta=[]
    for name in MODELS:
        d=ROOT/f'scripts/result/ss_experiments/malt_vs_non_malt/uni_v2_{name}'
        model=suite.SlideClassifier(name,1536,256,0.3).cuda().eval()
        model.load_state_dict(torch.load(d/'best_model.pth',map_location='cuda',weights_only=True),strict=True)
        models[name]=model
        modelmeta.append({'model':name,'parameter_count':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':sha(d/'best_model.pth'),
            'checkpoint_size_bytes':(d/'best_model.pth').stat().st_size,'configuration':json.loads((d/'config.json').read_text())})
    meta['models']=modelmeta
    (out/'runtime.json').write_text(json.dumps(meta,indent=2))
    old={name:pd.read_csv(DRAFT/f'supporting_data/internal_predictions_{name}.csv').set_index('study_slide_id') for name in MODELS}
    ext=pd.read_csv(DRAFT/'supporting_data/external_final_predictions.csv').set_index('study_slide_id')
    refrows=[];repeatrows=[];diagrows=[];index_archive={};inputrows=[];maxerr=0
    for count,row in enumerate(manifest.itertuples(),1):
        t0=time.monotonic()
        with h5py.File(row.feature_path,'r') as f: features=f['features'][:];coords=f['coords'][:]
        assert features.shape[1]==1536 and len(features)>0 and len(coords)==len(features)
        key={'scope':row.scope,'study_slide_id':row.study_slide_id,'label':int(row.label)}
        ids=pick(len(features),row.reference_seed,row.scope)
        index_archive[row.study_slide_id+'_reference']=ids
        batch={'features':torch.from_numpy(features[ids]).float().unsqueeze(0),'coords':torch.from_numpy(coords[ids]).long().unsqueeze(0)}
        with torch.inference_mode():
            hybrid_logit=None
            for name,model in models.items():
                logit=float(model(batch).item());score=float(torch.sigmoid(torch.tensor(logit)).item())
                expected=float(old[name].loc[row.study_slide_id,'probability']) if row.scope=='internal' else (float(ext.loc[row.study_slide_id,'malt_probability']) if name=='hybrid_evidence_mil' else None)
                err=None if expected is None else abs(score-expected)
                if expected is not None:
                    maxerr=max(maxerr,err)
                    assert err<=3e-5 and (score>=.5)==(expected>=.5),f'Reproduction mismatch: {row.study_slide_id}/{name}: {score} vs {expected}'
                refrows.append({**key,'model':name,'score':score,'logit':logit,'prediction_at_0_5':int(score>=.5),'saved_score':expected,'absolute_reproduction_error':err,'seed':row.reference_seed})
                if name=='hybrid_evidence_mil':hybrid_logit=logit
            diag=diagnostics(models['hybrid_evidence_mil'],batch)
            assert abs(diag['instrumented_logit']-hybrid_logit)<1e-5
            diagrows.append({**key,**diag})
            reps=2 if args.pilot else 20
            selected=[pick(len(features),repeat_seed(row.scope,row.study_slide_id,r),row.scope) for r in range(1,reps+1)]
            for start in range(0,reps,args.batch_size):
                chunk=selected[start:start+args.batch_size]
                batchrep={'features':torch.from_numpy(np.stack([features[ii] for ii in chunk])).float(),'coords':torch.from_numpy(np.stack([coords[ii] for ii in chunk])).long()}
                scores=torch.sigmoid(models['hybrid_evidence_mil'](batchrep)).cpu().numpy()
                for j,score in enumerate(scores,start+1):
                    index_archive[f'{row.study_slide_id}_repeat{j:02d}']=selected[j-1]
                    repeatrows.append({**key,'repeat_id':j,'score':float(score),'prediction_at_0_5':int(score>=.5),
                        'seed':repeat_seed(row.scope,row.study_slide_id,j),'n_available_rows':len(features),'n_unique_sampled_rows':len(np.unique(selected[j-1]))})
        inputrows.append({**key,'n_available_rows':len(features),'n_reference_unique_rows':len(np.unique(ids)),'feature_sha256':sha(row.feature_path),
                          'seconds_including_io_and_hash':time.monotonic()-t0})
        if count%10==0 or count==len(manifest):
            print(f'Completed {count}/{len(manifest)}; elapsed {time.monotonic()-started:.1f}s; max reproduced score error {maxerr:.3g}',flush=True)
            pd.DataFrame(refrows).to_csv(out/'reference_predictions.csv',index=False)
            pd.DataFrame(repeatrows).to_csv(out/'repeated_bag_predictions.csv',index=False)
            pd.DataFrame(diagrows).to_csv(out/'internal_signal_diagnostics.csv',index=False)
            pd.DataFrame(inputrows).to_csv(out/'input_integrity.csv',index=False)
    np.savez_compressed(out/'bag_indices.npz',**index_archive)
    meta.update({'completed_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.monotonic()-started,
        'n_slides':len(manifest),'n_reference_scores':len(refrows),'n_repeated_scores':len(repeatrows),
        'max_absolute_reproduction_error':maxerr,'max_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'complete':True})
    (out/'runtime.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:meta[k] for k in ['complete','wall_seconds','n_slides','n_reference_scores','n_repeated_scores','max_absolute_reproduction_error','max_gpu_allocated_mib']}),flush=True)

if __name__=='__main__':main()

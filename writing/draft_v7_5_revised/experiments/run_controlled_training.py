#!/usr/bin/env python
"""Nine predefined supplementary runs with fixed splits and matched bag schedules."""
from pathlib import Path
import copy,hashlib,json,os,random,time,sys,platform
from datetime import datetime,timezone
import h5py,numpy as np,pandas as pd,torch
from torch import nn
from sklearn.metrics import roc_auc_score
from run_checkpoint_analyses import ROOT,DRAFT,sha,prepare_manifest,pick
import ss_experiment_suite as suite
from summarize_checkpoint_analyses import metrics

OUT=Path(__file__).resolve().parent/'controlled_training'
SEEDS=[17,29,43]
VARIANTS=['matched_attention','fixed_scaling','learned_gate']

def seed32(text):return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4],'little')

class ConstantGate(nn.Module):
    def forward(self,x):return x.new_zeros((len(x),5))

class MatchedAttention(nn.Module):
    def __init__(self,template):
        super().__init__();self.project=copy.deepcopy(template.encoder.project);self.attention=copy.deepcopy(template.encoder.attention);self.classifier=copy.deepcopy(template.classifier)
    def forward(self,batch):
        h=self.project(batch['features']);z,_=self.attention(h);return self.classifier(z).squeeze(1)

def predict(model,rows,cache,bags):
    model.eval();records=[]
    with torch.inference_mode():
        for start in range(0,len(rows),8):
            chunk=rows.iloc[start:start+8]
            features=torch.stack([cache[sid][0][bags[sid]] for sid in chunk.study_slide_id])
            coords=torch.stack([cache[sid][1][bags[sid]] for sid in chunk.study_slide_id])
            scores=model({'features':features,'coords':coords}).sigmoid().cpu().numpy()
            for r,p in zip(chunk.itertuples(),scores):records.append({'scope':r.scope,'study_slide_id':r.study_slide_id,'label':int(r.label),'score':float(p)})
    return pd.DataFrame(records)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    OUT.mkdir(exist_ok=True);start=time.monotonic();torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    meta={'started_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(Path(__file__).with_name('CONTROLLED_TRAINING_PROTOCOL.md')),
          'runner_sha256':sha(__file__),'original_model_source_sha256':sha(ROOT/'scripts/ss_experiment_suite.py'),'seeds':SEEDS,'variants':VARIANTS,
          'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),'cuda':torch.version.cuda,
          'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),'gpu_index':3,'deterministic_algorithms':True,'complete':False}
    (OUT/'runtime.json').write_text(json.dumps(meta,indent=2))
    mapping=pd.read_csv(DRAFT/'qa/slide_code_mapping_PRIVATE.csv');imap=mapping[mapping.scope.eq('internal')].set_index('original_slide_id').study_slide_id.to_dict()
    original=ROOT/'scripts/result/ss_experiments/malt_vs_non_malt/uni_v2_hybrid_evidence_mil/cohort_split.csv'
    cohort=pd.read_csv(original);train=cohort[cohort.split.eq('train')].copy().reset_index(drop=True);train['study_slide_id']=train.slide_id.map(imap)
    assert len(train)==144 and train.label.sum()==67
    ev=prepare_manifest();inte=ev[ev.scope.eq('internal')].reset_index(drop=True);exte=ev[ev.scope.eq('external')].reset_index(drop=True)
    inputs=pd.concat([train[['study_slide_id','label','feature_path']],ev[['study_slide_id','label','feature_path']]],ignore_index=True)
    assert inputs.study_slide_id.nunique()==361
    cache={};evbags={};integrity=[]
    for r in inputs.itertuples():
        with h5py.File(r.feature_path,'r') as f:feat=f['features'][:];xy=f['coords'][:]
        cache[r.study_slide_id]=(torch.from_numpy(feat).float().cuda(),torch.from_numpy(xy).long().cuda())
        integrity.append({'study_slide_id':r.study_slide_id,'label':int(r.label),'feature_sha256':sha(r.feature_path),'n_rows':len(feat)})
    for r in ev.itertuples():evbags[r.study_slide_id]=torch.from_numpy(pick(len(cache[r.study_slide_id][0]),r.reference_seed,r.scope)).cuda()
    pd.DataFrame(integrity).to_csv(OUT/'input_integrity.csv',index=False)
    (OUT/'training_slide_order.json').write_text(json.dumps(train.study_slide_id.tolist()))
    trainlabels=torch.tensor(train.label.to_numpy(),dtype=torch.float32,device='cuda')
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([77/67],device='cuda'))
    summaries=[];predictions=[];batchchecks=[]
    print(f'Cached {len(cache)} feature files in {time.monotonic()-start:.1f}s; GPU allocated {torch.cuda.memory_allocated()/2**30:.2f}GiB',flush=True)
    for seed in SEEDS:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        template=suite.SlideClassifier('hybrid_evidence_mil',1536,256,.3)
        schedules=np.empty((30,144,512),dtype=np.int32);orders=np.empty((30,144),dtype=np.int32)
        for e in range(1,31):
            orders[e-1]=np.random.default_rng(seed32(f'controlled-v6|{seed}|{e}|order')).permutation(144)
            for i,r in enumerate(train.itertuples()):schedules[e-1,i]=pick(len(cache[r.study_slide_id][0]),seed32(f'controlled-v6|{seed}|{e}|{r.study_slide_id}'),'internal')
        np.savez_compressed(OUT/f'sampling_seed{seed}.npz',indices=schedules,orders=orders)
        schedules_gpu=torch.from_numpy(schedules.astype(np.int64)).cuda()
        for variant in VARIANTS:
            rstart=time.monotonic();rundir=OUT/f'{variant}_seed{seed}';rundir.mkdir(exist_ok=True)
            assert not (rundir/'best_model.pth').exists(),f'Refusing overwrite {rundir}'
            if variant=='matched_attention':model=MatchedAttention(template)
            else:
                model=copy.deepcopy(template)
                if variant=='fixed_scaling':model.encoder.gate=ConstantGate()
            common=(model.project,model.attention,model.classifier) if variant=='matched_attention' else (model.encoder.project,model.encoder.attention,model.classifier)
            for target,source in zip(common,[template.encoder.project,template.encoder.attention,template.classifier]):
                assert all(torch.equal(v,source.state_dict()[k]) for k,v in target.state_dict().items())
            model=model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-4)
            bestauc=-1;bestepoch=0;history=[]
            for e in range(1,31):
                model.train();losses=[]
                for bi,s in enumerate(range(0,144,8)):
                    ix=orders[e-1,s:s+8];ids=train.iloc[ix].study_slide_id.tolist()
                    feat=torch.stack([cache[sid][0][schedules_gpu[e-1,int(i)]] for i,sid in zip(ix,ids)])
                    xy=torch.stack([cache[sid][1][schedules_gpu[e-1,int(i)]] for i,sid in zip(ix,ids)])
                    torch.manual_seed(seed32(f'controlled-v6|{seed}|{e}|{bi}|dropout'))
                    optimizer.zero_grad(set_to_none=True)
                    logits=model({'features':feat,'coords':xy});loss=lossfn(logits,trainlabels[torch.as_tensor(ix,device='cuda',dtype=torch.long)])
                    if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {variant}/{seed}/{e}/{bi}')
                    loss.backward()
                    if e==1 and bi==0:
                        g={k:float(p.grad.norm()) if p.grad is not None else None for k,p in model.named_parameters()}
                        (rundir/'first_batch_gradient_norms.json').write_text(json.dumps(g,indent=2))
                        assert all(v is not None and np.isfinite(v) for v in g.values()),'Disconnected or invalid parameter gradient'
                    optimizer.step();losses.append(float(loss.item()))
                result=predict(model,inte,cache,evbags);mm=metrics(result.label.to_numpy(),result.score.to_numpy())
                history.append({'epoch':e,'loss':np.mean(losses),**mm})
                if mm['auroc']>bestauc:
                    bestauc=mm['auroc'];bestepoch=e;torch.save(model.state_dict(),rundir/'best_model.pth')
                if e%10==0:print(f'{variant} seed{seed} epoch{e}/30 best internal AUC {bestauc:.4f} elapsed {time.monotonic()-rstart:.1f}s',flush=True)
            torch.save(model.state_dict(),rundir/'final_model.pth');pd.DataFrame(history).to_csv(rundir/'history.csv',index=False)
            finalmetric=history[-1]
            model.load_state_dict(torch.load(rundir/'best_model.pth',map_location='cuda',weights_only=True))
            for scope,rows in [('internal',inte),('external',exte)]:
                result=predict(model,rows,cache,evbags);result['variant']=variant;result['seed']=seed;result['selected_epoch']=bestepoch
                result.to_csv(rundir/f'{scope}_predictions.csv',index=False);predictions.append(result)
                summaries.append({'variant':variant,'seed':seed,'scope':scope,'selected_epoch':bestepoch,'parameters':sum(p.numel() for p in model.parameters()),
                                  **metrics(result.label.to_numpy(),result.score.to_numpy()),'final_epoch_internal_auroc':finalmetric['auroc'],
                                  'wall_seconds_run':time.monotonic()-rstart,'checkpoint_sha256':sha(rundir/'best_model.pth')})
            pd.DataFrame(summaries).to_csv(OUT/'run_metrics.csv',index=False);pd.concat(predictions,ignore_index=True).to_csv(OUT/'all_predictions.csv',index=False)
            print(f'COMPLETED {variant} seed{seed} selectedepoch{bestepoch} runseconds{time.monotonic()-rstart:.1f}',flush=True)
            del model,optimizer
        del template,schedules_gpu
    df=pd.DataFrame(summaries);summary=[];diffs=[]
    for (variant,scope),g in df.groupby(['variant','scope'],sort=False):
        row={'variant':variant,'scope':scope,'n_seeds':3,'parameters':int(g.parameters.iloc[0])}
        for met in ['auroc','accuracy','sensitivity','specificity','balanced_accuracy']:
            row[met+'_mean']=g[met].mean();row[met+'_sd']=g[met].std(ddof=1);row[met+'_min']=g[met].min();row[met+'_max']=g[met].max()
        summary.append(row)
    for scope,g in df.groupby('scope',sort=False):
        p=g.pivot(index='seed',columns='variant',values='auroc')
        for other in ['matched_attention','fixed_scaling']:
            d=p.learned_gate-p[other];diffs.append({'scope':scope,'comparison':'learned_gate minus '+other,'difference_mean':d.mean(),'difference_sd':d.std(ddof=1),**{f'seed{s}':d.loc[s] for s in SEEDS}})
    pd.DataFrame(summary).to_csv(OUT/'configuration_summary.csv',index=False);pd.DataFrame(diffs).to_csv(OUT/'paired_seed_differences.csv',index=False)
    meta.update({'complete':True,'completed_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.monotonic()-start,
                 'max_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'n_completed_runs':9,'n_prediction_rows':len(pd.concat(predictions))})
    (OUT/'runtime.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python
"""Bounded, internal-selected algorithm study. Physical GPU 3 only."""
from pathlib import Path
import argparse, copy, hashlib, json, os, platform, random, sys, time
from datetime import datetime, timezone
import h5py, joblib, numpy as np, pandas as pd, torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'draft_v7_focused_narrative'
sys.path.insert(0, str(OLD / 'experiments'))
from run_checkpoint_analyses import ROOT, prepare_manifest, pick, sha
from summarize_checkpoint_analyses import metrics
from models import build_model

SEEDS = [17,29,43]
EXTRA_SEEDS = [59,71]
VARIANTS = ['hybrid_control','attention_control','hybrid_l2','attention_l2','attention_mask',
            'direct_attention','subspace_attention','subspace_largebatch','hybrid_twobag','hybrid_consistency']
CONTROLS = ['hybrid_control','attention_control']

def stamp(): return datetime.now(timezone.utc).isoformat()
def seed32(text): return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')
def writejson(path, obj): path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
def configuration(v):
    return {'variant':v, 'epochs':30, 'batch_size':32 if v=='subspace_largebatch' else 8,
            'learning_rate':3e-4 if v=='subspace_largebatch' else 5e-5, 'weight_decay':1e-4,
            'positive_weight':77/67, 'two_views':v in ['hybrid_twobag','hybrid_consistency'],
            'consistency_weight':.1 if v=='hybrid_consistency' else 0,
            'evaluation_subspace_windows':21 if v.startswith('subspace') else 1}

class Study:
    def __init__(self, out):
        self.out=out; out.mkdir(exist_ok=True); (out/'PRIVATE').mkdir(exist_ok=True)
        mapping=pd.read_csv(OLD/'qa/slide_code_mapping_PRIVATE.csv')
        imap=mapping[mapping.scope.eq('internal')].set_index('original_slide_id').study_slide_id.to_dict()
        cohort=pd.read_csv(ROOT/'scripts/result/ss_experiments/malt_vs_non_malt/uni_v2_hybrid_evidence_mil/cohort_split.csv')
        self.train=cohort[cohort.split.eq('train')].copy().reset_index(drop=True)
        self.train['study_slide_id']=self.train.slide_id.map(imap);self.train['scope']='train'
        self.ev=prepare_manifest()
        self.internal=self.ev[self.ev.scope.eq('internal')].reset_index(drop=True)
        self.external=self.ev[self.ev.scope.eq('external')].reset_index(drop=True)
        assert len(self.train)==144 and self.train.label.sum()==67 and self.train.study_slide_id.notna().all()
        inputs=pd.concat([self.train[['scope','study_slide_id','label','feature_path']],self.ev[['scope','study_slide_id','label','feature_path']]],ignore_index=True)
        inputs.to_csv(out/'PRIVATE/input_manifest.csv',index=False)
        inputs.drop(columns='feature_path').to_csv(out/'cohort_manifest.csv',index=False)
        self.inputs=inputs;self.cache={};self.integrity=[];self.reference={};self.runs=[];self.predictions=[]
        self.load(inputs[inputs.scope.ne('external')])
        self.labels=torch.tensor(self.train.label.to_numpy(),dtype=torch.float32,device='cuda')
        self.lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([77/67],device='cuda'))
        self.schedule_cache={};self.linear={}
        self.prepare_reference(self.internal)

    def load(self,rows):
        for r in rows.itertuples():
            if r.study_slide_id in self.cache:continue
            with h5py.File(r.feature_path,'r') as f:feat=f['features'][:];xy=f['coords'][:]
            assert feat.ndim==2 and feat.shape[1]==1536 and len(feat)>0 and len(xy)==len(feat)
            assert np.isfinite(feat).all()
            self.cache[r.study_slide_id]=(torch.from_numpy(feat).float().cuda(),torch.from_numpy(xy).long().cuda())
            self.integrity.append({'study_slide_id':r.study_slide_id,'scope':r.scope,'label':int(r.label),'n_rows':len(feat),'feature_sha256':sha(r.feature_path)})
        pd.DataFrame(self.integrity).to_csv(self.out/'input_integrity.csv',index=False)

    def prepare_reference(self,rows):
        for r in rows.itertuples():
            self.reference[r.study_slide_id]=torch.from_numpy(pick(len(self.cache[r.study_slide_id][0]),r.reference_seed,r.scope)).cuda()

    def batch(self,ids,indices):
        return {'features':torch.stack([self.cache[s][0][idx] for s,idx in zip(ids,indices)]),
                'coords':torch.stack([self.cache[s][1][idx] for s,idx in zip(ids,indices)])}

    def predict(self,model,rows,bags=None):
        model.eval();result=[];bags=self.reference if bags is None else bags
        with torch.inference_mode():
            for start in range(0,len(rows),8):
                chunk=rows.iloc[start:start+8];ids=chunk.study_slide_id.tolist()
                logits=model(self.batch(ids,[bags[s] for s in ids])).cpu().numpy()
                scores=torch.sigmoid(torch.from_numpy(logits)).numpy()
                for r,z,p in zip(chunk.itertuples(),logits,scores):
                    result.append({'scope':r.scope,'study_slide_id':r.study_slide_id,'label':int(r.label),'logit':float(z),'score':float(p)})
        return pd.DataFrame(result)

    def schedule(self,seed):
        if seed in self.schedule_cache:return self.schedule_cache[seed]
        first=np.empty((30,144,512),np.int32);second=np.empty_like(first);orders=np.empty((30,144),np.int32)
        for e in range(1,31):
            orders[e-1]=np.random.default_rng(seed32(f'controlled-v6|{seed}|{e}|order')).permutation(144)
            for i,r in enumerate(self.train.itertuples()):
                n=len(self.cache[r.study_slide_id][0])
                first[e-1,i]=pick(n,seed32(f'controlled-v6|{seed}|{e}|{r.study_slide_id}'),'internal')
                second[e-1,i]=pick(n,seed32(f'algorithm-v8-view2|{seed}|{e}|{r.study_slide_id}'),'internal')
        if seed in SEEDS:
            prior=np.load(OLD/f'experiments/controlled_training/sampling_seed{seed}.npz')
            assert np.array_equal(first,prior['indices']) and np.array_equal(orders,prior['orders'])
        np.savez_compressed(self.out/f'sampling_seed{seed}.npz',indices=first,second_indices=second,orders=orders)
        result=(torch.from_numpy(first.astype(np.int64)).cuda(),torch.from_numpy(second.astype(np.int64)).cuda(),orders)
        self.schedule_cache[seed]=result
        return result

    def train_run(self,variant,seed,stage):
        cfg=configuration(variant);rdir=self.out/f'{variant}_seed{seed}'
        assert not rdir.exists(),f'Refusing overwrite {rdir}'
        rdir.mkdir();writejson(rdir/'configuration.json',{**cfg,'seed':seed,'stage':stage})
        start=time.monotonic();first,second,orders=self.schedule(seed)
        model=build_model(variant,seed).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        initial_hash=hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        best=-1;bestepoch=0;history=[]
        for e in range(1,31):
            model.train();losses=[];bces=[];kls=[]
            for bi,startidx in enumerate(range(0,144,cfg['batch_size'])):
                ix=orders[e-1,startidx:startidx+cfg['batch_size']];ids=self.train.iloc[ix].study_slide_id.tolist()
                batch=self.batch(ids,[first[e-1,int(i)] for i in ix])
                targets=self.labels[torch.as_tensor(ix,device='cuda',dtype=torch.long)]
                torch.manual_seed(seed32(f'controlled-v6|{seed}|{e}|{bi}|dropout'))
                opt.zero_grad(set_to_none=True);z1=model(batch);bce=self.lossfn(z1,targets);kl=z1.new_zeros(())
                if cfg['two_views']:
                    b2=self.batch(ids,[second[e-1,int(i)] for i in ix])
                    torch.manual_seed(seed32(f'controlled-v6|{seed}|{e}|{bi}|dropout-view2'))
                    z2=model(b2);bce=.5*(bce+self.lossfn(z2,targets))
                    kl=.5*((z1.sigmoid()-z2.sigmoid())*(z1-z2)).mean()
                loss=bce+cfg['consistency_weight']*kl
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {variant}/{seed}/{e}/{bi}')
                loss.backward()
                if e==1 and bi==0:
                    grads={k:None if p.grad is None else float(p.grad.norm()) for k,p in model.named_parameters()}
                    writejson(rdir/'first_batch_gradient_norms.json',grads)
                    assert all(v is not None and np.isfinite(v) for v in grads.values())
                opt.step();losses.append(float(loss));bces.append(float(bce));kls.append(float(kl))
            pred=self.predict(model,self.internal);mm=metrics(pred.label.to_numpy(),pred.score.to_numpy())
            history.append({'epoch':e,'loss':float(np.mean(losses)),'bce':float(np.mean(bces)),'symmetric_kl':float(np.mean(kls)),**mm})
            if mm['auroc']>best:
                best=mm['auroc'];bestepoch=e;torch.save(model.state_dict(),rdir/'best_model.pth')
            if e%10==0:print(f'{variant} seed {seed}: epoch {e}/30, best internal AUC {best:.4f}, elapsed {time.monotonic()-start:.1f}s',flush=True)
        torch.save(model.state_dict(),rdir/'final_model.pth')
        pd.DataFrame(history).to_csv(rdir/'history.csv',index=False)
        model.load_state_dict(torch.load(rdir/'best_model.pth',map_location='cuda',weights_only=True))
        pred=self.predict(model,self.internal);mm=metrics(pred.label.to_numpy(),pred.score.to_numpy())
        pred['variant']=variant;pred['seed']=seed;pred['stage']=stage;pred['selected_epoch']=bestepoch;pred['inference']='reference_bag'
        pred.to_csv(rdir/'internal_predictions.csv',index=False)
        record={'variant':variant,'seed':seed,'stage':stage,'selected_epoch':bestepoch,'scope':'internal',
                'inference':'reference_bag','parameters':sum(p.numel() for p in model.parameters()),
                'optimizer_steps':30*int(np.ceil(144/cfg['batch_size'])),'initial_state_tensor_sha256':initial_hash,
                'checkpoint_sha256':sha(rdir/'best_model.pth'),'wall_seconds_training_including_eval':time.monotonic()-start,**mm}
        if variant in CONTROLS and seed in SEEDS:
            oldvariant={'hybrid_control':'learned_gate','attention_control':'matched_attention'}[variant]
            prior=pd.read_csv(OLD/f'experiments/controlled_training/{oldvariant}_seed{seed}/internal_predictions.csv').set_index('study_slide_id')
            pp=pred.set_index('study_slide_id').loc[prior.index]
            error=float(np.max(np.abs(pp.score-prior.score)))
            assert error<=3e-5 and int(prior.selected_epoch.iloc[0])==bestepoch,(variant,seed,error,bestepoch)
            assert np.array_equal(pp.score.ge(.5),prior.score.ge(.5))
            writejson(rdir/'historical_control_reproduction.json',{'status':'PASS','maximum_score_error':error,'selected_epoch':bestepoch})
        self.runs.append(record);self.predictions.append(pred)
        self.flush();print(f'DONE {variant} seed {seed}, internal AUC {mm["auroc"]:.4f}, {record["wall_seconds_training_including_eval"]:.1f}s',flush=True)
        del model,opt

    def flush(self):
        if self.runs:pd.DataFrame(self.runs).to_csv(self.out/'run_metrics.csv',index=False)
        if self.predictions:pd.concat(self.predictions,ignore_index=True).to_csv(self.out/'all_predictions.csv',index=False)

    def fit_linear(self):
        start=time.monotonic();saved={};trainy=self.train.label.to_numpy()
        for variant in ['mean_logistic','l2mean_logistic']:
            x=np.stack([self.representation(s,variant,None) for s in self.train.study_slide_id])
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',solver='lbfgs',max_iter=3000,tol=1e-8,random_state=20261104))
            model.fit(x,trainy);assert model[-1].n_iter_.max()<3000
            rdir=self.out/variant;rdir.mkdir();joblib.dump(model,rdir/'model.joblib');np.save(rdir/'train_representations.npy',x)
            writejson(rdir/'training.json',{'C':1.,'class_weight':'balanced','training_slide_ids':self.train.study_slide_id.tolist(),
                      'training_rows':'all available feature rows per slide','scaler_fit_scope':'144 training slides only','iterations':model[-1].n_iter_.tolist()})
            self.linear[variant]=model
            for inference in ['reference_bag','full_feature_mean']:
                p=self.linear_predict(variant,self.internal,inference)
                p.to_csv(rdir/f'internal_{inference}_predictions.csv',index=False)
                self.predictions.append(p)
                self.runs.append({'variant':variant,'seed':-1,'stage':'deterministic_control','selected_epoch':0,'scope':'internal','inference':inference,
                                  'parameters':1537,'checkpoint_sha256':sha(rdir/'model.joblib'),**metrics(p.label.to_numpy(),p.score.to_numpy())})
        self.flush();print(f'Deterministic mean-feature controls fitted in {time.monotonic()-start:.1f}s',flush=True)

    def representation(self,sid,variant,indices):
        x=self.cache[sid][0]
        if indices is not None:x=x[indices]
        if variant=='l2mean_logistic':x=torch.nn.functional.normalize(x,p=2,dim=-1,eps=1e-12)*np.sqrt(1536)
        return x.mean(dim=0).cpu().numpy()

    def linear_predict(self,variant,rows,inference):
        x=np.stack([self.representation(s,variant,self.reference[s] if inference=='reference_bag' else None) for s in rows.study_slide_id])
        scores=self.linear[variant].predict_proba(x)[:,1]
        p=rows[['scope','study_slide_id','label']].copy();p['score']=scores;p['variant']=variant;p['seed']=-1;p['stage']='deterministic_control';p['selected_epoch']=0;p['inference']=inference
        return p

    def select(self):
        d=pd.DataFrame(self.runs);d=d[(d.stage=='screen')&(~d.variant.isin(CONTROLS))]
        ranking=[]
        for variant,g in d.groupby('variant'):
            assert set(g.seed)==set(SEEDS)
            ranking.append({'variant':variant,'mean_internal_auroc':float(g.auroc.mean()),'sd_internal_auroc':float(g.auroc.std(ddof=1)),
                            'parameters':int(g.parameters.iloc[0]),'per_seed_auroc':{str(int(x.seed)):float(x.auroc) for x in g.itertuples()}})
        ranking.sort(key=lambda r:(-round(r['mean_internal_auroc'],12),r['sd_internal_auroc'],r['parameters'],r['variant']))
        winner=ranking[0]['variant']
        lock={'frozen_utc':stamp(),'selection_scope':'internal 49 slides only; no new external inference yet',
              'winner':winner,'ranking':ranking,'screen_seeds':SEEDS,'additional_seeds':EXTRA_SEEDS,
              'additional_variants':['hybrid_control',winner],'protocol_sha256':sha(HERE/'PROTOCOL.md'),
              'runner_sha256':sha(__file__),'models_sha256':sha(HERE/'models.py'),
              'screen_run_metrics_sha256':sha(self.out/'run_metrics.csv'),
              'screen_checkpoint_hashes':{f'{r["variant"]}_seed{r["seed"]}':r['checkpoint_sha256'] for r in self.runs if r['stage']=='screen'}}
        assert len(lock['screen_checkpoint_hashes'])==30
        writejson(self.out/'selection_lock.json',lock)
        pd.DataFrame(self.runs).to_csv(self.out/'screen_run_metrics_frozen.csv',index=False)
        return winner

    def external_evaluation(self,winner):
        lockpath=self.out/'selection_lock.json';assert lockpath.exists()
        lockhash=sha(lockpath)
        writejson(self.out/'external_evaluation_start.json',{'started_utc':stamp(),'selection_lock_sha256':lockhash,
                  'neural_checkpoints_frozen':{f'{r["variant"]}_seed{r["seed"]}':r['checkpoint_sha256'] for r in self.runs if r['stage'] in ['screen','confirmation']}})
        self.load(self.inputs[self.inputs.scope.eq('external')]);self.prepare_reference(self.external)
        neural=[r.copy() for r in self.runs if r['stage'] in ['screen','confirmation']]
        for rec in neural:
            v,seed=rec['variant'],rec['seed'];rdir=self.out/f'{v}_seed{seed}'
            assert sha(rdir/'best_model.pth')==rec['checkpoint_sha256']
            model=build_model(v,seed).cuda();model.load_state_dict(torch.load(rdir/'best_model.pth',map_location='cuda',weights_only=True))
            p=self.predict(model,self.external);p['variant']=v;p['seed']=seed;p['stage']=rec['stage'];p['selected_epoch']=rec['selected_epoch'];p['inference']='reference_bag'
            p.to_csv(rdir/'external_predictions.csv',index=False);self.predictions.append(p)
            self.runs.append({**rec,'scope':'external',**metrics(p.label.to_numpy(),p.score.to_numpy())})
            del model
        for v in self.linear:
            for inference in ['reference_bag','full_feature_mean']:
                p=self.linear_predict(v,self.external,inference)
                p.to_csv(self.out/v/f'external_{inference}_predictions.csv',index=False);self.predictions.append(p)
                self.runs.append({'variant':v,'seed':-1,'stage':'deterministic_control','selected_epoch':0,'scope':'external','inference':inference,
                                  'parameters':1537,'checkpoint_sha256':sha(self.out/v/'model.joblib'),**metrics(p.label.to_numpy(),p.score.to_numpy())})
        assert sha(lockpath)==lockhash
        self.flush();print('All locked external reference-bag evaluations complete.',flush=True)
        self.repeat_evaluation(winner)

    def repeat_evaluation(self,winner):
        bags={};archive={}
        for rep in range(1,6):
            bags[rep]={}
            for r in self.ev.itertuples():
                seed=seed32(f'algorithm-v8-eval|{r.scope}|{r.study_slide_id}|{rep}')
                ids=pick(len(self.cache[r.study_slide_id][0]),seed,r.scope)
                archive[f'{r.study_slide_id}_repeat{rep}']=ids;bags[rep][r.study_slide_id]=torch.from_numpy(ids).cuda()
        archive.update({f'{s}_reference':ids.cpu().numpy() for s,ids in self.reference.items()})
        np.savez_compressed(self.out/'evaluation_bag_indices.npz',**archive)
        reps=[];summ=[]
        for v in ['hybrid_control',winner]:
            for seed in SEEDS+EXTRA_SEEDS:
                rdir=self.out/f'{v}_seed{seed}';model=build_model(v,seed).cuda()
                model.load_state_dict(torch.load(rdir/'best_model.pth',map_location='cuda',weights_only=True))
                for rep in range(1,6):
                    for rows in [self.internal,self.external]:
                        p=self.predict(model,rows,bags[rep]);p['variant']=v;p['seed']=seed;p['repeat_id']=rep;reps.append(p)
                del model
                print(f'Five-bag evaluation complete: {v} seed {seed}',flush=True)
        d=pd.concat(reps,ignore_index=True);d.to_csv(self.out/'repeated_bag_predictions.csv',index=False)
        avg=d.groupby(['variant','seed','scope','study_slide_id','label'],as_index=False).score.mean()
        avg['inference']='five_bag_probability_mean';avg.to_csv(self.out/'five_bag_mean_predictions.csv',index=False)
        for (v,seed,scope),g in avg.groupby(['variant','seed','scope']):
            summ.append({'variant':v,'seed':seed,'scope':scope,'inference':'five_bag_probability_mean',**metrics(g.label.to_numpy(),g.score.to_numpy())})
        pd.DataFrame(summ).to_csv(self.out/'five_bag_metrics.csv',index=False)

    def pilot(self):
        result=[]
        for v in ['hybrid_control','subspace_attention','hybrid_consistency']:
            seed=17;first,second,orders=self.schedule(seed);model=build_model(v,seed).cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=5e-5);ix=orders[0,:8];ids=self.train.iloc[ix].study_slide_id.tolist()
            model.train();torch.cuda.synchronize();t=time.monotonic()
            for j in range(3):
                opt.zero_grad(set_to_none=True);z=model(self.batch(ids,[first[0,int(i)] for i in ix]));loss=self.lossfn(z,self.labels[torch.as_tensor(ix,device='cuda',dtype=torch.long)])
                if v=='hybrid_consistency':
                    z2=model(self.batch(ids,[second[0,int(i)] for i in ix]));loss=.5*(loss+self.lossfn(z2,self.labels[torch.as_tensor(ix,device='cuda',dtype=torch.long)]))+.05*((z.sigmoid()-z2.sigmoid())*(z-z2)).mean()
                loss.backward();opt.step()
            torch.cuda.synchronize();trainsec=(time.monotonic()-t)/3;t=time.monotonic()
            self.predict(model,self.internal);torch.cuda.synchronize();evalsec=time.monotonic()-t
            result.append({'variant':v,'train_seconds_per_batch':trainsec,'internal_eval_seconds':evalsec,'estimated_30_epoch_seconds':30*(18*trainsec+evalsec)})
            del model,opt
        writejson(self.out/'pilot_timing.json',result);print(json.dumps(result,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    out=HERE/('pilot' if args.pilot else 'outputs');assert not (out/'runtime.json').exists(),'Existing run directory: do not overwrite'
    out.mkdir(exist_ok=True);t=time.monotonic()
    meta={'started_utc':stamp(),'CUDA_VISIBLE_DEVICES':'3','gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),'gpu_name':torch.cuda.get_device_name(0),
          'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),'cuda':torch.version.cuda,
          'protocol_sha256':sha(HERE/'PROTOCOL.md'),'runner_sha256':sha(__file__),'models_sha256':sha(HERE/'models.py'),
          'original_model_sha256':sha(ROOT/'scripts/ss_experiment_suite.py'),'memory_fraction_limit':.15,'complete':False,'pilot':args.pilot}
    writejson(out/'runtime.json',meta)
    try:
        study=Study(out);print(f'Cached training/internal features in {time.monotonic()-t:.1f}s; {torch.cuda.memory_allocated()/2**30:.2f}GiB allocated',flush=True)
        if args.pilot:study.pilot()
        else:
            for seed in SEEDS:
                for v in VARIANTS:study.train_run(v,seed,'screen')
            study.fit_linear();winner=study.select();print(f'INTERNAL-ONLY SELECTION LOCKED: {winner}',flush=True)
            for seed in EXTRA_SEEDS:
                for v in ['hybrid_control',winner]:study.train_run(v,seed,'confirmation')
            study.external_evaluation(winner)
            assert len([r for r in study.runs if r['stage'] in ['screen','confirmation'] and r['scope']=='internal'])==34
            meta['selected_variant']=winner;meta['n_neural_runs']=34;meta['n_linear_controls']=2
        meta.update({'complete':True,'completed_utc':stamp(),'wall_seconds':time.monotonic()-t,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
        writejson(out/'runtime.json',meta);print(json.dumps(meta,indent=2),flush=True)
    except Exception as ex:
        meta.update({'failed_utc':stamp(),'wall_seconds':time.monotonic()-t,'error':repr(ex)})
        writejson(out/'runtime.json',meta);raise

if __name__=='__main__':main()

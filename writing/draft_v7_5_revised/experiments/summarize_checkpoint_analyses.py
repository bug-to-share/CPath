#!/usr/bin/env python
"""Prespecified, reproducible statistics for fixed-checkpoint supplementary outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, norm
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

BASE=Path(__file__).resolve().parent
OUT=BASE/'outputs'
HYBRID='hybrid_evidence_mil'

def wilson(k,n):
    if not n:return np.nan,np.nan
    z=norm.ppf(.975);p=k/n;den=1+z*z/n
    return ((p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den,
            (p+z*z/(2*n)+z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den)

def metrics(y,p):
    tn,fp,fn,tp=map(int,confusion_matrix(y,p>=.5,labels=[0,1]).ravel())
    r={'n':len(y),'n_malt':int(sum(y)),'n_ss':int(len(y)-sum(y)),'auroc':roc_auc_score(y,p),'average_precision':average_precision_score(y,p),
       'accuracy':(tn+tp)/len(y),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2,
       'tn':tn,'fp':fp,'fn':fn,'tp':tp,'brier':float(np.mean((p-y)**2))}
    for key,k,n in [('accuracy',tn+tp,len(y)),('sensitivity',tp,tp+fn),('specificity',tn,tn+fp)]:
        r[key+'_ci_low'],r[key+'_ci_high']=wilson(k,n)
    return r

def main():
    runtime=json.loads((OUT/'runtime.json').read_text());assert runtime['complete'] and runtime['n_slides']==217
    ref=pd.read_csv(OUT/'reference_predictions.csv');rep=pd.read_csv(OUT/'repeated_bag_predictions.csv')
    assert len(ref)==868 and len(rep)==4340
    assert rep.groupby('study_slide_id').size().eq(20).all()
    assert ref.groupby('study_slide_id').model.nunique().eq(4).all()
    mrows=[]
    for (scope,model),df in ref.groupby(['scope','model'],sort=False):mrows.append({'scope':scope,'model':model,**metrics(df.label.to_numpy(),df.score.to_numpy())})
    metricdf=pd.DataFrame(mrows)
    external=ref[ref.scope.eq('external')].pivot(index='study_slide_id',columns='model',values='score')
    y=ref[(ref.scope.eq('external'))&(ref.model.eq(HYBRID))].set_index('study_slide_id').loc[external.index,'label'].to_numpy()
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);rng=np.random.default_rng(20261104);nboot=10000
    # The same resampled slides are used for all models. Multinomial counts are
    # algebraically identical to sampling each class with replacement.
    wp=rng.multinomial(len(pos),np.ones(len(pos))/len(pos),size=nboot)/len(pos)
    wn=rng.multinomial(len(neg),np.ones(len(neg))/len(neg),size=nboot)/len(neg)
    draws={}
    for model in external:
        p=external[model].to_numpy();pairs=(p[pos,None]>p[None,neg]).astype(float)+.5*(p[pos,None]==p[None,neg])
        assert abs(pairs.mean()-roc_auc_score(y,p))<1e-12
        draws[model]=np.sum((wp@pairs)*wn,axis=1)
        lo,hi=np.quantile(draws[model],[.025,.975])
        mask=metricdf.scope.eq('external')&metricdf.model.eq(model)
        metricdf.loc[mask,'auroc_ci_low']=lo;metricdf.loc[mask,'auroc_ci_high']=hi
    comparisons=[]
    hp=external[HYBRID].to_numpy();hc=(hp>=.5)==y
    for model in external:
        if model==HYBRID:continue
        p=external[model].to_numpy();oc=(p>=.5)==y
        diff=draws[HYBRID]-draws[model];lo,hi=np.quantile(diff,[.025,.975]);a=int(sum(hc&~oc));b=int(sum(~hc&oc))
        comparisons.append({'scope':'external','comparison':'Hybrid minus '+model,'auroc_difference':roc_auc_score(y,hp)-roc_auc_score(y,p),
            'difference_ci_low':lo,'difference_ci_high':hi,'hybrid_only_correct':a,'other_only_correct':b,
            'mcnemar_exact_p_exploratory':binomtest(a,a+b,.5).pvalue if a+b else 1.0,'n_bootstrap':nboot,
            'note':'Whole-model comparisons; unadjusted descriptive intervals and exploratory tests, not confirmatory family-wise inference.'})
    metricdf.to_csv(OUT/'reference_model_metrics.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'external_paired_comparisons.csv',index=False)
    reference=ref[ref.model.eq(HYBRID)].set_index('study_slide_id')
    per=[]
    for (scope,sid,label),df in rep.groupby(['scope','study_slide_id','label'],sort=False):
        p=df.score.to_numpy();rp=float(reference.loc[sid,'score'])
        per.append({'scope':scope,'study_slide_id':sid,'label':label,'reference_score':rp,'score_mean':p.mean(),'score_sd':p.std(ddof=1),
            'score_min':p.min(),'score_max':p.max(),'score_range':np.ptp(p),'positive_draws':int(sum(p>=.5)),
            'crosses_threshold_0_5':int(p.min()<.5<=p.max()),'draws_disagreeing_with_reference':int(sum((p>=.5)!=(rp>=.5))),
            'mean_prediction_at_0_5':int(p.mean()>=.5),'n_available_rows':int(df.n_available_rows.iloc[0]),
            'unique_sampled_rows_mean':df.n_unique_sampled_rows.mean(),'unique_sampled_rows_min':int(df.n_unique_sampled_rows.min()),'unique_sampled_rows_max':int(df.n_unique_sampled_rows.max())})
    per=pd.DataFrame(per);per.to_csv(OUT/'sampling_per_slide.csv',index=False)
    drawmetrics=[]
    for (scope,r),df in rep.groupby(['scope','repeat_id'],sort=False):drawmetrics.append({'scope':scope,'repeat_id':r,**metrics(df.label.to_numpy(),df.score.to_numpy())})
    dm=pd.DataFrame(drawmetrics);dm.to_csv(OUT/'sampling_per_draw_metrics.csv',index=False)
    summary=[]
    for scope,df in per.groupby('scope',sort=False):
        d=dm[dm.scope.eq(scope)];r={'scope':scope,'n':len(df),'n_draws':20,'score_sd_median':df.score_sd.median(),
        'score_sd_q25':df.score_sd.quantile(.25),'score_sd_q75':df.score_sd.quantile(.75),'score_sd_max':df.score_sd.max(),
        'score_range_median':df.score_range.median(),'threshold_crossing_slides':int(df.crosses_threshold_0_5.sum()),
        'threshold_crossing_fraction':df.crosses_threshold_0_5.mean(),'mean_reference_disagreement_fraction':df.draws_disagreeing_with_reference.mean()/20,
        'draw_auroc_mean':d.auroc.mean(),'draw_auroc_sd':d.auroc.std(ddof=1),'draw_auroc_min':d.auroc.min(),'draw_auroc_max':d.auroc.max()}
        for label,name in [(0,'ss'),(1,'malt')]:
            a=df[df.label.eq(label)];r['n_'+name]=len(a);r['threshold_crossing_'+name]=int(a.crosses_threshold_0_5.sum())
        r.update({'mean_bag_'+k:v for k,v in metrics(df.label.to_numpy(),df.score_mean.to_numpy()).items()})
        summary.append(r)
    pd.DataFrame(summary).to_csv(OUT/'sampling_summary.csv',index=False)
    diagnostics=pd.read_csv(OUT/'internal_signal_diagnostics.csv')
    diag_summary=[]
    for scope,df in diagnostics.groupby('scope',sort=False):
        score=reference.loc[df.study_slide_id,'score'].to_numpy()
        r={'scope':scope,'n':len(df),'positive_margin_fraction_mean':df.positive_margin_fraction.mean(),
           'margin_mean_min':df.margin_mean.min(),'margin_mean_max':df.margin_mean.max(),
           'margin_mean_vs_score_spearman':spearmanr(df.margin_mean,score).statistic,
           'margin_mean_vs_binary_label_spearman':spearmanr(df.margin_mean,df.label).statistic}
        for name in ['attention','mean','std','high_margin','low_margin_plus_region']:
            for pref in ['gate_','norm_','scaled_norm_']:
                x=df[pref+name];r[pref+name+'_mean']=x.mean();r[pref+name+'_sd']=x.std(ddof=1)
            r['gate_'+name+'_min']=df['gate_'+name].min();r['gate_'+name+'_max']=df['gate_'+name].max()
        r['mean_gate_largest_count']=int(df[[c for c in df if c.startswith('gate_')]].idxmax(axis=1).eq('gate_mean').sum())
        diag_summary.append(r)
    pd.DataFrame(diag_summary).to_csv(OUT/'signal_diagnostic_summary.csv',index=False)
    qa={'reference_scores':len(ref),'repeated_scores':len(rep),'all_original_reference_checks_pass':bool(ref.absolute_reproduction_error.dropna().le(3e-5).all()),
        'max_reproduction_error':ref.absolute_reproduction_error.max(),'all_repeat_seeds_unique':rep.seed.nunique()==len(rep),
        'no_repeat_scores_missing':bool(rep.score.notna().all()),'bag_index_arrays':len(np.load(OUT/'bag_indices.npz').files),
        'comparisons':comparisons,'sampling':summary,'diagnostics':diag_summary,'runtime_wall_seconds':runtime['wall_seconds']}
    assert qa['all_repeat_seeds_unique'] and qa['bag_index_arrays']==4557
    (OUT/'analysis_summary.json').write_text(json.dumps(qa,indent=2))
    print(metricdf.to_string(index=False));print(pd.DataFrame(comparisons).to_string(index=False))
    print(pd.DataFrame(summary)[['scope','n','score_sd_median','threshold_crossing_slides','threshold_crossing_malt','threshold_crossing_ss','draw_auroc_min','draw_auroc_max','mean_bag_auroc']].to_string(index=False))

if __name__=='__main__':main()

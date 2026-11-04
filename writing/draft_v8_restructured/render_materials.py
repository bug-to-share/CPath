"""Regenerate v8 tables and vector figures from frozen, publication-coded data."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/trident-v8-mpl')
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

R=Path(__file__).resolve().parent; H=R/'source_data/historical'; A=R/'source_data/algorithm'
F=R/'figures';T=R/'tables'
BLUE='#386A8D';ORANGE='#B8703E';TEAL='#39796C';GRAY='#818990';INK='#263746'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.6,'axes.titlesize':9.2,
 'axes.labelsize':8.6,'xtick.labelsize':8.2,'ytick.labelsize':8.2,'legend.fontsize':8.1,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,
 'svg.fonttype':'none','pdf.fonttype':42,'lines.linewidth':1.5,'savefig.facecolor':'white'})
def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(F/f'{name}.{ext}',dpi=450)
    plt.close(fig)
def panel(ax,letter,title):
    ax.set_title(title,loc='left',pad=12)
    ax.text(-.16,1.07,letter,transform=ax.transAxes,fontsize=11,fontweight='bold',va='bottom')
def esc(s):return str(s).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
def tex_table(name,caption,label,headers,rows,note='',widths=None,small=True):
    cols=('l'+'r'*(len(headers)-1)) if widths is None else ''.join(r'>{\raggedright\arraybackslash}p{'+str(w)+r'\textwidth}' for w in widths)
    txt=r'\begin{table}[tbp]\centering'+'\n'+r'\caption{'+caption+'}'+r'\label{'+label+'}\n'
    if small:txt+=r'\small'+'\n'
    txt+=r'\begin{tabular}{'+cols+'}\n'+r'\toprule'+'\n'
    txt+=' & '.join(r'\textbf{'+x+'}' for x in headers)+r' \\'+'\n'+r'\midrule'+'\n'
    txt+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n'+r'\bottomrule\end{tabular}'+'\n'
    if note:txt+='\n'+r'\par\medskip\footnotesize '+note+'\n'
    txt+=r'\end{table}'+'\n';(T/(name+'.tex')).write_text(txt)
def long_table(name,caption,headers,rows,widths,note=''):
    cols=''.join(r'>{\raggedright\arraybackslash}p{'+str(w)+r'\textwidth}' for w in widths)
    head=' & '.join(r'\textbf{'+x+'}' for x in headers)+r' \\'+'\n'
    s=r'{\small\begin{longtable}{'+cols+'}\n'+r'\caption{'+caption+r'}\\'+'\n'+r'\toprule'+'\n'+head+r'\midrule\endfirsthead'+'\n'+r'\toprule'+'\n'+head+r'\midrule\endhead'+'\n'
    s+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n'+r'\bottomrule\end{longtable}}'+'\n'
    if note:s+=r'{\footnotesize '+note+'}\n'
    (T/(name+'.tex')).write_text(s)

internal=pd.read_csv(H/'internal_model_metrics.csv')
mi=internal.set_index('aggregator').loc['hybrid_evidence_mil'].to_dict()
me=json.loads((H/'external_final_metrics.json').read_text())
ext=pd.read_csv(H/'external_final_predictions.csv')
pred=pd.read_csv(H/'reference_predictions.csv')
five=pd.read_csv(A/'selected_control_all_five_per_seed.csv')
confirm=pd.read_csv(A/'confirmation_summary.csv')
sampling=pd.read_csv(H/'sampling_per_draw_metrics.csv')
review=json.loads((R/'source_data/review/numeric_audit.json').read_text())

def tables():
    rows=[['Slides, n','144','49','168'],['MALT / SS, n','67 / 77','23 / 26','152 / 16'],
      ['Role','Training','Checkpoint selection','External evaluation'],
      ['Patients, n',r'\blankfield{H01}',r'\blankfield{H01}',r'\blankfield{H01}'],
      ['Slides per patient',r'\blankfield{H01}',r'\blankfield{H01}',r'\blankfield{H01}'],
      ['Age and sex',r'\blankfield{H01}',r'\blankfield{H01}',r'\blankfield{H01}'],
      ['Gland site; specimen type',r'\blankfield{H01}',r'\blankfield{H01}',r'\blankfield{H01}'],
      ['Collection dates; clinical source',r'\blankfield{H01}',r'\blankfield{H01}',r'\blankfield{H01}'],
      ['Confirmed stain',r'\blankfield{H01}',r'\blankfield{H01}',r'\blankfield{H01}'],
      ['Diagnostic reference standard',r'\blankfield{H02}',r'\blankfield{H02}',r'\blankfield{H02}']]
    tex_table('cohort','Study cohorts and clinical fields reserved for source records','tab:cohort',
      ['Characteristic','Training','Internal validation','External'],rows,
      'Dashes with completion codes are unfilled hospital-record fields, not zero counts or evidence that the records do not exist. Patient linkage and specimen-level attributes must be incorporated before a patient-level interpretation. All available counts refer to WSIs.',[.34,.18,.20,.18])
    def ci(d,k,pct=False):
        f=lambda x:f'{100*x:.2f}'+r'\%' if pct else f'{x:.4f}'
        return f'{f(d[k])} ({f(d[k+"_ci_low"])}--{f(d[k+"_ci_high"])})'
    rows=[[n,ci(mi,k,p),ci(me,k,p)] for n,k,p in [('AUROC','auroc',False),('AP','auprc',False),
       ('Sensitivity','sensitivity',True),('Specificity','specificity',True),('Accuracy','accuracy',True),('PPV','ppv',True),('NPV','npv',True)]]
    rows.append(['Balanced accuracy',r'88.21\%',r'81.09\%'])
    tex_table('primary','Retained Hybrid performance with conditional 95\% intervals','tab:primary',
      ['Metric','Internal validation','External evaluation'],rows,
      'Threshold metrics use 0.5. AUROC and AP intervals use 2,000 class-stratified WSI bootstrap draws; proportion intervals use Wilson\'s method. These intervals condition on fitted predictions and do not adjust for selection or patient clustering.',[.22,.34,.34])
    def rowmean(df,v,sc):return df[(df.variant==v)&(df.scope==sc)].iloc[0]
    rows=[]
    for v,n in [('hybrid_control','Hybrid'),('attention_l2','Attention+L2')]:
      g=confirm[(confirm.variant==v)&(confirm.seed_group=='all_five')]
      x=rowmean(g,v,'internal');y=rowmean(g,v,'external')
      rows.append([n,f'{x.auroc_mean:.4f} $\\pm$ {x.auroc_sd:.4f}',f'{y.auroc_mean:.4f} $\\pm$ {y.auroc_sd:.4f}',f'{100*y.sensitivity_mean:.2f}'+r'\%',f'{100*y.specificity_mean:.2f}'+r'\%'])
    tex_table('controlled','Five-seed comparison of Hybrid and the internally selected candidate','tab:controlled',
      ['Model','Validation AUROC','External AUROC','Ext. sens.','Ext. spec.'],rows,
      'AUROCs are means $\pm$ sample SD across seeds 17, 29, 43, 59 and 71; sensitivity and specificity are means of seed-specific rates at 0.5. These are not ensemble metrics. The same 49 validation and 168 external slides were reused. The three-seed selection was fixed before the additional seeds and new external inference.',[.19,.23,.23,.12,.12])
    names={'hybrid_clam_rrt_ensemble':'Three-model ensemble','hybrid_evidence_mil':'Hybrid','abmil':'ABMIL','mean':'Mean pooling',
      'clam_sb':'Local attention pooling (CLAM-style)','dsmil':'Local dual-stream pooling','dtfd':'Local two-tier attention',
      'transmil':'Local transformer pooling','rrtmil':'Local grouped-region transformer','proto_mil':'Local prototype pooling',
      'dist_proto_mil':'Distribution--prototype pooling','titan':'CONCH v1.5 + TITAN'}
    rows=[]
    for _,r in internal.iterrows():
      rows.append([names.get(r.aggregator,esc(r.model)),f'{r.auroc:.4f}',f'{r.auprc:.4f}',f'{100*r.sensitivity:.2f}',f'{100*r.specificity:.2f}'])
    long_table('S1_historical','Historical complete-implementation comparison on 49 validation slides',
      ['Implementation','AUROC','AP','Sens. (\%)','Spec. (\%)'],rows,[.44,.10,.10,.12,.12],
      'Threshold 0.5. This table retains all historical models, not a ranking of official full-method reproductions. TITAN uses a different encoder and instance budget. Code-ID mappings and original metrics are in Source Data. The ensemble combines Hybrid, local CLAM-style pooling and the grouped-region transformer.')
    pd.DataFrame([{'code_id':k,'display_name':v} for k,v in names.items()]).to_csv(R/'source_data/implementation_names.csv',index=False)
    cmp=pd.read_csv(H/'external_paired_comparisons.csv')
    # A compact complete-checkpoint table; exact paired intervals remain in copied source CSV.
    rm=pd.read_csv(H/'reference_model_metrics.csv');rows=[]
    for _,r in rm[rm.scope=='external'].iterrows():
      rows.append([names.get(r.model,esc(r.model)),f'{r.auroc:.4f}',f'{100*r.sensitivity:.2f}',f'{100*r.specificity:.2f}',f'{int(r.tp)}/{int(r.fn)}/{int(r.tn)}/{int(r.fp)}'])
    long_table('S2_external','External complete-checkpoint comparison on identical reference bags',
      ['Implementation','AUROC','Sens. (\%)','Spec. (\%)',r'TP/FN/\allowbreak TN/FP'],rows,[.39,.11,.13,.13,.15])
    c=pd.read_csv(R/'source_data/controlled/configuration_summary.csv');rows=[]
    for v,n in [('matched_attention','Matched attention'),('fixed_scaling','Full fixed scaling'),('learned_gate','Full learned gate')]:
      i=rowmean(c,v,'internal');e=rowmean(c,v,'external')
      rows.append([n,str(int(i.parameters)),f'{i.auroc_mean:.4f} $\\pm$ {i.auroc_sd:.4f}',f'{e.auroc_mean:.4f} $\\pm$ {e.auroc_sd:.4f}'])
    long_table('S3_matched','Three-seed controlled training comparison', ['Configuration','Parameters','Validation AUROC','External AUROC'],rows,[.27,.17,.23,.23],
      'Seeds 17, 29 and 43. Means $\pm$ sample SD. The algorithm screen reran the attention and learned-gate controls and reproduced them; those reruns are not new independent seeds.')
    screen=pd.read_csv(A/'screen_summary.csv'); labels={'hybrid_control':'Hybrid control','attention_control':'Matched attention','hybrid_l2':'Hybrid + L2','attention_l2':'Attention + L2','attention_mask':'Attention masking','direct_attention':'Original-space attention','subspace_attention':'Feature-subspace attention','subspace_largebatch':'Subspace, batch32/lr3e-4','hybrid_twobag':'Hybrid, two bags','hybrid_consistency':'Hybrid, two bags + KL'}
    rows=[]
    for v,n in labels.items():
      i=rowmean(screen,v,'internal');e=rowmean(screen,v,'external')
      rows.append([n,f'{i.auroc_mean:.4f} $\\pm$ {i.auroc_sd:.4f}',f'{e.auroc_mean:.4f} $\\pm$ {e.auroc_sd:.4f}'])
    long_table('S4_screen','All ten configurations in the three-seed algorithm screen',['Configuration','Validation AUROC','External AUROC'],rows,[.43,.23,.23])
    run=pd.read_csv(A/'original_run_metrics.csv');run=run[run.stage!='deterministic_control'];rows=[]
    for (v,s),g in run.groupby(['variant','seed'],sort=False):
      i=g[g.scope=='internal'].iloc[0];e=g[g.scope=='external'].iloc[0]
      rows.append([labels[v],int(s),int(i.selected_epoch),f'{i.auroc:.4f}',f'{e.auroc:.4f}',f'{100*e.sensitivity:.2f}',f'{100*e.specificity:.2f}'])
    long_table('S5_all_runs','All 34 neural training runs: selected checkpoint results',
      ['Configuration','Seed','Epoch','Val. AUC','Ext. AUC','Ext. sens.','Ext. spec.'],rows,[.31,.06,.07,.10,.10,.11,.11],
      'Sensitivities and specificities are percentages at threshold 0.5. Every planned run is retained. Full epoch histories and checkpoints remain in the experiment archive.')
    linear=pd.read_csv(A/'linear_controls.csv');rows=[]
    for _,r in linear.iterrows():
      rows.append(['L2 mean' if r.variant=='l2mean_logistic' else 'Raw mean', '512 patches' if r.inference=='reference_bag' else 'Full features',r.scope,f'{r.auroc:.4f}',f'{100*r.sensitivity:.2f}',f'{100*r.specificity:.2f}'])
    long_table('S6_linear','Deterministic linear controls',['Features','Input','Cohort','AUROC','Sens. (\%)','Spec. (\%)'],rows,[.15,.18,.14,.11,.14,.14])
    ens=pd.read_csv(A/'ensemble_metrics.csv');rows=[]
    for _,r in ens[ens.variant.isin(['hybrid_control','attention_l2']) & ens['mode'].isin(['three_seed_reference','five_seed_reference','five_seed_five_bag'])].iterrows():
      mode={'three_seed_reference':'3 seeds; ref. bag','five_seed_reference':'5 seeds; ref. bag','five_seed_five_bag':'5 seeds; 5 bags'}[r['mode']]
      rows.append([labels[r.variant],mode,r.scope,f'{r.auroc:.4f}',f'{100*r.sensitivity:.2f}',f'{100*r.specificity:.2f}'])
    long_table('S7_ensembles','Probability ensembles of the complete seed sets',['Model','Averaging','Cohort','AUROC','Sens. (\%)','Spec. (\%)'],rows,[.20,.24,.13,.10,.11,.11],
      'Reference bags and the five newly sampled bags are separate inputs. Each displayed metric is calculated after probability averaging; it is not an average of AUROCs.')
    ss=pd.read_csv(H/'sampling_summary.csv');rows=[]
    for _,r in ss.iterrows():
      rows.append([r.scope,f'{r.score_sd_median:.4f}',str(int(r.threshold_crossing_slides))+'/'+str(int(r.n)),f'{r.draw_auroc_min:.4f}--{r.draw_auroc_max:.4f}',f'{r.mean_bag_auroc:.4f}'])
    long_table('S8_sampling','Twenty-bag sensitivity analysis of the retained Hybrid checkpoint',['Cohort','Median score SD','Crossing 0.5','Draw AUROC range','Mean-score AUROC'],rows,[.13,.18,.17,.24,.17])
    cases=pd.read_csv(H/'interpretation_probability_comparison.csv');rows=[]
    for _,r in cases.iterrows():
      rows.append([esc(r.publication_case),r.study_slide_id,f'{r.main_evaluation_probability:.4f}',f'{r.interpretation_probability:.4f}',f'{r.absolute_probability_difference:.4f}'])
    long_table('S9_regions','Linked performance and visualization outputs',['Case','Slide code','Performance score','Visualization score','Absolute difference'],rows,[.15,.15,.21,.21,.19])
    (R/'qa/table_generation.json').write_text(json.dumps({'main_tables':3,'supplement_tables':9,'neural_run_rows':len(run)//2,'source_only':True},indent=2))

def fig1():
    fig=plt.figure(figsize=(6.54,5.8));ax=fig.add_axes([.035,.035,.93,.93]);ax.axis('off');ax.set(xlim=(-1,101),ylim=(0,100))
    def box(x,y,w,h,s,fc='#EAF0F5',size=8.5):
      ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.35,rounding_size=.8',facecolor=fc,edgecolor='#91A3AF',lw=.8))
      ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=size,color=INK)
    def ar(x,y,u,v):ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=9,lw=.9,color=GRAY))
    ax.text(0,100,'a  Cohort flow',weight='bold',fontsize=11,va='top')
    box(0,76,45,17,'Internal cohort: 193 slides\n90 MALT | 103 SS')
    box(55,82,45,11,'External collection: 227 slides','#F5ECE4')
    box(55,64,45,11,'Successful predictions: 223\n4 without a prediction','#F5ECE4')
    box(55,44,45,13,'Analysis set: 168 slides\n152 MALT | 16 SS\n55 predicted slides not included','#F5ECE4')
    ar(77.5,82,77.5,75);ar(77.5,64,77.5,57)
    box(0,51,21,17,'Training: 144\n67 MALT\n77 SS')
    box(25,51,20,17,'Validation: 49\n23 MALT\n26 SS')
    ar(10,76,10,68);ar(35,76,35,68)
    ax.text(35,45,'Checkpoint\nselection',ha='center',va='top',fontsize=8.4)
    ax.plot([0,100],[35,35],color='#D3DBE0',lw=.8)
    ax.text(0,31,'b  Shared image-to-score workflow',weight='bold',fontsize=11)
    texts=['Tissue\nsegmentation','20× patches\n512 × 512 px','Frozen UNI2-h\n1,536 features','512 sampled\npatch vectors','Aggregation\nMALT score']
    for j,s in enumerate(texts):
      x=j*20.5;box(x,12,18,13,s,fc='#EAF1ED',size=8.2)
      if j<4:ar(x+18,18.5,x+20.5,18.5)
    ax.text(50,3,'Slide-label supervision • fixed encoder • retrospective evaluation',ha='center',fontsize=8.5,color=INK)
    save(fig,'Figure_1_cohorts')

def fig2():
    fig,axs=plt.subplots(2,2,figsize=(6.54,6.3));fig.subplots_adjust(left=.1,right=.97,bottom=.08,top=.94,wspace=.44,hspace=.58)
    for j,scope in enumerate(['internal','external']):
      ax=axs[0,j]
      for model,name,col in [('hybrid_evidence_mil','Hybrid',BLUE),('abmil','ABMIL',GRAY)]:
        g=pred[(pred.scope==scope)&(pred.model==model)];f,t,_=roc_curve(g.label,g.score)
        ax.plot(f,t,color=col,label=f'{name}: {roc_auc_score(g.label,g.score):.3f}')
      ax.plot([0,1],[0,1],'--',color='#C4C7CA',lw=.8);ax.set(xlim=(0,1.01),ylim=(0,1.03),xlabel='False-positive rate',ylabel='True-positive rate');ax.legend(loc='lower right')
      panel(ax,'ab'[j],['Internal validation','External evaluation'][j])
      ax=axs[1,j];g=pred[(pred.scope==scope)&(pred.model=='hybrid_evidence_mil')];cm=confusion_matrix(g.label,g.score>=.5);prop=cm/cm.sum(axis=1,keepdims=True)
      ax.imshow(prop,cmap='Blues',vmin=0,vmax=1)
      for (i,k),n in np.ndenumerate(cm):ax.text(k,i,f'{n}\n({100*prop[i,k]:.1f}%)',ha='center',va='center',fontsize=10,color='white' if prop[i,k]>.6 else INK)
      ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['SS','MALT'],yticklabels=['SS','MALT'],xlabel='Predicted class',ylabel='Reference class');ax.tick_params(length=0)
      panel(ax,'cd'[j],['Hybrid: 49 slides','Hybrid: 168 slides'][j])
    save(fig,'Figure_2_primary')

def fig3():
    fig,axs=plt.subplots(2,1,figsize=(6.54,6.6),gridspec_kw={'height_ratios':[1.25,1]});fig.subplots_adjust(left=.36,right=.97,top=.94,bottom=.17,hspace=.53)
    labels=[('SS (slide series)','SS; stain unspecified (n=16)'),('H&E-like','MALT; H&E-like tag (n=33)'),('H&E','MALT; HE directory (n=8)'),('IHC / special stain','MALT; IHC/special tag (n=111)')]
    ax=axs[0];rng=np.random.default_rng(884)
    for j,(tag,name) in enumerate(labels):
      g=ext[ext.stain_group==tag];col=TEAL if tag=='SS (slide series)' else BLUE
      ax.scatter(g.malt_probability,j+rng.uniform(-.17,.17,len(g)),s=12,alpha=.6,c=col,edgecolors='none')
      med=g.malt_probability.median();ax.plot([med,med],[j-.26,j+.26],c=INK,lw=1.6)
    ax.axvline(.5,color=GRAY,lw=.9,ls='--');ax.set(yticks=range(4),yticklabels=[x[1] for x in labels],xlabel='Retained Hybrid MALT score',xlim=(-.03,1.03),ylim=(-.55,3.55));ax.invert_yaxis();panel(ax,'a','Scores by archive category')
    ax=axs[1];allm=review['external_main'];sub=review['archive_tag_sensitivity_analysis_not_confirmed_HE_validation']
    for j,(k,l) in enumerate([('auroc','AUROC'),('sensitivity','Sensitivity'),('specificity','Specificity'),('balanced_accuracy','Balanced accuracy')]):
      ax.plot([allm[k],sub[k]],[j,j],c='#C5CDD1',lw=1.5)
      ax.scatter(allm[k],j-.08,color=BLUE,s=32,label='All 168 slides' if j==0 else None)
      ax.scatter(sub[k],j+.08,color=ORANGE,s=32,label='57-slide descriptive subset' if j==0 else None)
    ax.set(yticks=range(4),yticklabels=['AUROC','Sensitivity','Specificity','Balanced accuracy'],xlim=(.55,1.02),ylim=(-.55,3.55),xlabel='Metric value (threshold 0.5 for rates)');ax.invert_yaxis();panel(ax,'b','After omitting IHC/special archive tags')
    handles,legend_labels=ax.get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.60,.015),ncol=1,fontsize=8.1)
    save(fig,'Figure_3_material')

def fig4():
    fig,axs=plt.subplots(2,2,figsize=(6.54,6.1));fig.subplots_adjust(left=.11,right=.97,bottom=.10,top=.94,wspace=.44,hspace=.60)
    seeds=[17,29,43,59,71]
    for j,scope in enumerate(['internal','external']):
      ax=axs[0,j];ax.axvspan(2.5,4.5,color='#F0F2F4');ax.axvline(2.5,c='#B5BBC0',ls='--',lw=.8)
      g=five[five.scope==scope]
      for k,s in enumerate(seeds):
        a=float(g[(g.variant=='hybrid_control')&(g.seed==s)].auroc.iloc[0]);b=float(g[(g.variant=='attention_l2')&(g.seed==s)].auroc.iloc[0])
        ax.plot([k-.1,k+.1],[a,b],c='#ADB8C0',lw=1)
        ax.scatter(k-.1,a,color=BLUE,s=26,label='Hybrid' if k==0 else None);ax.scatter(k+.1,b,color=ORANGE,s=26,label='Attention+L2' if k==0 else None)
      ax.set(xticks=range(5),xticklabels=seeds,xlabel='Training seed',ylabel='AUROC',xlim=(-.5,4.5),ylim=(.55,1.01));ax.legend(loc='lower left',fontsize=8.1)
      panel(ax,'ab'[j],['Training: validation','Training: external'][j])
      ax=axs[1,j];g=sampling[sampling.scope==scope];ref=mi['auroc'] if scope=='internal' else me['auroc']
      ax.plot(g.repeat_id,g.auroc,c=TEAL,lw=.8);ax.scatter(g.repeat_id,g.auroc,color=TEAL,s=14);ax.axhline(ref,c=GRAY,ls='--',lw=1)
      ax.set(xlabel='Patch draw (fixed checkpoint)',ylabel='AUROC',xlim=(.5,20.5),ylim=(.55,1.01),xticks=[1,5,10,15,20]);panel(ax,'cd'[j],['Input sampling: validation','Input sampling: external'][j])
      ax.text(.04,.1,f'Range: {g.auroc.min():.3f}–{g.auroc.max():.3f}',transform=ax.transAxes,fontsize=8.2)
    save(fig,'Figure_4_stability')

if __name__=='__main__':
    tables();fig1();fig2();fig3();fig4()
    (R/'qa/figure_contract.json').write_text(json.dumps({'backend':'Python/matplotlib','width_inches':6.54,'font_size_min_points':8.1,'formats':['PDF','SVG','450dpi PNG'],'quantitative_source':'frozen CSV/JSON only','main_figures':4,'histology_pixels':'unaltered inherited PDF in supplement'},indent=2)+'\n')
    print('Generated 4 main figures, 3 main tables and 9 supplementary tables.')

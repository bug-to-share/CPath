"""Draw manuscript comparisons from archived metrics; no training or inference."""
from pathlib import Path
import os, json
os.environ.setdefault('MPLCONFIGDIR', '/tmp/trident_v75_visual_mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

R = Path(__file__).resolve().parent
D, A = R/'supporting_data', R/'development_analyses/analysis'
F, S = R/'figures', R/'analysis/table_visualization'
S.mkdir(exist_ok=True)
B, O, T, P, G, INK = '#336B91', '#B46B37', '#397D72', '#77618E', '#7B838B', '#263945'
CMAP = LinearSegmentedColormap.from_list('metric_blue', ['#F7FAFC', '#E4EEF4', '#C5DCE7', '#81AFC6', '#356F90'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.7,'axes.titlesize':8.5,
 'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7.3,'legend.fontsize':7,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'legend.frameon':False,'lines.linewidth':1,'svg.fonttype':'none','pdf.fonttype':42,
 'savefig.facecolor':'white','axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':G})
names={'hybrid_control':'Hybrid control','attention_control':'Matched attention',
 'hybrid_l2':'Normalized Hybrid','attention_l2':'Normalized attention',
 'attention_mask':'Attention masking','direct_attention':'Direct attention',
 'subspace_attention':'Subspace attention','subspace_largebatch':'Subspace + large batch',
 'hybrid_twobag':'Hybrid, two bags','hybrid_consistency':'Hybrid + consistency',
 'mean_logistic':'Raw mean + logistic','l2mean_logistic':'L2 mean + logistic'}
metrics=['auroc','auprc','accuracy','sensitivity','specificity','f1']
labels=['AUROC','AP','Accuracy','Sensitivity','Specificity','F1']
saved=[]
def save(fig, stem):
    for ext in ['pdf','svg','png']:
        fig.savefig(F/f'{stem}.{ext}',dpi=450,bbox_inches='tight',pad_inches=.07)
    saved.append(stem);plt.close(fig)
def title(ax, letter, text):
    ax.set_title(text,loc='left',pad=12)
    ax.text(-.10,1.055,letter,transform=ax.transAxes,weight='bold',fontsize=10,va='bottom')
def clean(ax):
    ax.set_axisbelow(True);ax.grid(axis='x',color='#E4E9ED',lw=.55)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
def matrix(ax, values, headers, rate_cols, rows=None, fontsize=6.8, vmin=0):
    ax.imshow(values,cmap=CMAP,vmin=vmin,vmax=1,aspect='auto',interpolation='nearest')
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v=values[i,j];s=f'{100*v:.2f}' if j in rate_cols else f'{v:.4f}'
            ax.text(j,i,s,ha='center',va='center',fontsize=fontsize,color='white' if (v-vmin)/(1-vmin)>.74 else INK)
    ax.set_xticks(range(len(headers)),headers);ax.xaxis.tick_top()
    ax.tick_params(axis='x',length=0,pad=6,labelsize=6.5)
    ax.set_yticks(range(len(values)),rows if rows is not None else ['']*len(values));ax.tick_params(axis='y',length=0)
    ax.set_xticks(np.arange(-.5,values.shape[1],1),minor=True)
    ax.set_yticks(np.arange(-.5,values.shape[0],1),minor=True)
    ax.grid(which='minor',color='white',lw=1.2);ax.tick_params(which='minor',length=0)
    for sp in ax.spines.values():sp.set_visible(False)

def internal():
    m=pd.read_csv(D/'internal_model_metrics.csv')
    assert len(m)==12
    m.to_csv(S/'figure2_internal_metrics.csv',index=False)
    fig=plt.figure(figsize=(7.2,4.75))
    ax=fig.add_axes([.295,.18,.215,.70]);hm=fig.add_axes([.55,.18,.435,.70])
    for i,row in m.iterrows():
        color=B if row.aggregator=='hybrid_evidence_mil' else (O if row.aggregator=='titan' else G)
        marker='s' if row.aggregator=='hybrid_clam_rrt_ensemble' else ('D' if row.aggregator=='titan' else 'o')
        ax.errorbar(row.auroc,i,xerr=[[row.auroc-row.auroc_ci_low],[row.auroc_ci_high-row.auroc]],fmt=marker,color=color,ms=4,capsize=2,lw=.9)
    ax.set(ylim=(11.5,-.5),yticks=range(12),yticklabels=m.model,xlim=(.58,1.02),xticks=[.6,.8,1],xlabel='AUROC (95% interval)')
    for tick,agg in zip(ax.get_yticklabels(),m.aggregator):
        if agg=='hybrid_evidence_mil':tick.set_weight('bold');tick.set_color(B)
        if agg=='titan':tick.set_color(O)
    clean(ax);title(ax,'a','Discrimination')
    matrix(hm,m[metrics].to_numpy(),['AUROC','AP','Acc. (%)','Sens. (%)','Spec. (%)','F1'],{2,3,4},vmin=.5)
    title(hm,'b','Metric profile')
    for a in [ax,hm]:a.axhline(10.5,color=O,ls=':',lw=1)
    ax.set_title('Discrimination',loc='left',y=1.10,pad=10)
    hm.set_title('Metric profile',loc='left',y=1.10,pad=10)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(.5,1),cmap=CMAP),cax=fig.add_axes([.64,.075,.27,.018]),orientation='horizontal',ticks=[.5,.75,1])
    cb.set_label('Cell color: metric value (0.5–1)',fontsize=6.7,labelpad=2)
    fig.text(.015,.052,'49 validation WSIs also selected checkpoints.\nOrange: different encoder and patch budget.',fontsize=6.8)
    save(fig,'Figure_2_metric_profiles')

def external():
    internal=pd.read_csv(D/'internal_model_metrics.csv').set_index('aggregator').loc['hybrid_evidence_mil']
    external=json.loads((D/'external_final_metrics.json').read_text())
    keys=['auroc','auprc','accuracy','sensitivity','specificity','ppv','npv','f1']
    labs=['AUROC','Average precision','Accuracy','Sensitivity','Specificity','PPV','NPV','F1']
    records=[]
    fig=plt.figure(figsize=(7.2,6.5));ax=fig.add_axes([.18,.55,.55,.35])
    for cohort,row,col,off in [('Internal',internal,B,-.14),('External',external,O,.14)]:
        for j,k in enumerate(keys):
            v,lo,hi=[float(row[z]) for z in [k,k+'_ci_low',k+'_ci_high']]
            records.append(dict(cohort=cohort,metric=k,value=v,ci_low=lo,ci_high=hi))
            ax.errorbar(v,j+off,xerr=[[v-lo],[hi-v]],fmt='o',ms=4,color=col,lw=1,capsize=2)
            ax.text(1.08 if cohort=='Internal' else 1.29,j,f'{v:.4f}',va='center',ha='center',fontsize=7,color=col,clip_on=False)
    ax.text(1.08,-1,'Internal',ha='center',color=B,fontsize=7)
    ax.text(1.29,-1,'External',ha='center',color=O,fontsize=7)
    ax.set(ylim=(7.6,-.65),yticks=range(8),yticklabels=labs,xlim=(0,1.02),xticks=np.arange(0,1.01,.2),xlabel='Estimate and 95% interval (0–1 scale)');clean(ax)
    title(ax,'a','Historical Hybrid: internal and external estimates')
    pd.DataFrame(records).to_csv(S/'figure3_cohort_intervals.csv',index=False)
    e=pd.read_csv(D/'external_final_predictions.csv');groups=[]
    for lab,cohort,tag in [('MALT: H&E-like','MALT_BJ','H&E-like'),('MALT: IHC/special','MALT_BJ','IHC / special stain'),('MALT: HE archive','MALT_HE',None),('SS: stain unknown','SS_BJ',None)]:
        z=e[e.cohort.eq(cohort)]
        if tag is not None:z=z[z.stain_group.eq(tag)]
        correct=int((z.label==z.prediction_at_0_5).sum());groups.append(dict(group=lab,n=len(z),correct=correct,incorrect=len(z)-correct))
    ax=fig.add_axes([.23,.17,.30,.24]);title(ax,'b','Archive-tag groups')
    for j,g in enumerate(groups):
        fraction=g['correct']/g['n'];ax.barh(j,fraction,color=B,height=.55);ax.barh(j,1-fraction,left=fraction,color='#E1C6B0',height=.55)
        ax.text(.98,j,f"{g['correct']}/{g['n']}",ha='right',va='center',color='white' if fraction>.96 else INK,fontsize=7)
    ax.set(yticks=range(4),yticklabels=[x['group'] for x in groups],ylim=(3.7,-.65),xlim=(0,1),xticks=[0,.5,1],xlabel='Fraction correctly classified');clean(ax)
    ax.legend(handles=[Line2D([0],[0],color=B,lw=5,label='Correct'),Line2D([0],[0],color='#E1C6B0',lw=5,label='Incorrect')],loc='upper center',bbox_to_anchor=(.4,-.26),ncol=2,fontsize=6.5)
    pd.DataFrame(groups).to_csv(S/'figure3_archive_groups.csv',index=False)
    audit=json.loads((R/'analysis/review_numeric_audit.json').read_text());rows=[]
    ax=fig.add_axes([.73,.17,.25,.24]);title(ax,'c','Post hoc tag exclusion')
    kk=['auroc','sensitivity','specificity','balanced_accuracy']
    for j,k in enumerate(kk):
        full=audit['external_main'][k];sub=audit['archive_tag_sensitivity_analysis_not_confirmed_HE_validation'][k]
        ax.plot([full,sub],[j-.09,j+.09],color='#BBC2C8',lw=1.5)
        for v,col,cohort,off in [(full,O,'Full external',-.09),(sub,P,'After tag exclusion',.09)]:
            ax.scatter(v,j+off,s=22,color=col,zorder=3);rows.append(dict(cohort=cohort,metric=k,value=v))
    ax.set(yticks=range(4),yticklabels=['AUROC','Sensitivity','Specificity','Balanced acc.'],ylim=(3.7,-.65),xlim=(0,1),xticks=[0,.5,1],xlabel='Metric value');clean(ax)
    ax.legend(handles=[Line2D([0],[0],marker='o',ls='',color=O,label='Full (168 WSIs)'),Line2D([0],[0],marker='o',ls='',color=P,label='Subset (57 WSIs)')],loc='upper center',bbox_to_anchor=(.40,-.26),fontsize=6.5)
    pd.DataFrame(rows).to_csv(S/'figure3_tag_exclusion.csv',index=False)
    fig.text(.015,.012,'Archive tags are provisional. The retained subset is not confirmed H&E-only validation.',fontsize=6.7)
    save(fig,'Figure_3_cohort_and_material')

def stability():
    neural=pd.read_csv(A/'selected_control_all_five_per_seed.csv');linear=pd.read_csv(A/'linear_controls.csv')
    linear=linear[linear.inference.eq('reference_bag')]
    allrows=pd.concat([neural,linear],ignore_index=True);allrows.to_csv(S/'figure4_seed_and_linear_metrics.csv',index=False)
    variants=['hybrid_control','attention_l2','mean_logistic','l2mean_logistic']
    cols=[B,T,G,P];fig=plt.figure(figsize=(7.2,5.4));summary=[]
    specs=[('a','internal','auroc',[.28,.58,.30,.30],'Internal AUROC'),('b','external','auroc',[.66,.58,.30,.30],'External AUROC'),('c','external','sensitivity',[.28,.13,.30,.30],'External sensitivity'),('d','external','specificity',[.66,.13,.30,.30],'External specificity')]
    for letter,scope,metric,pos,lab in specs:
        ax=fig.add_axes(pos)
        for i,(v,col) in enumerate(zip(variants,cols)):
            z=allrows[allrows.variant.eq(v)&allrows.scope.eq(scope)].sort_values('seed')
            vals=z[metric].to_numpy();offsets=np.linspace(-.18,.18,len(z)) if len(z)>1 else [0]
            for (_,row),off in zip(z.iterrows(),offsets):
                ax.scatter(row[metric],i+off,s=24 if len(z)==1 else 17,marker='D' if len(z)==1 else ('o' if row.seed in [17,29,43] else 's'),facecolor=col if row.seed in [17,29,43,-1] else 'white',edgecolor=col,lw=.8,zorder=3)
            mean=float(vals.mean());sd=float(vals.std(ddof=1)) if len(vals)>1 else None
            if len(vals)>1:
                if metric=='auroc':ax.errorbar(mean,i,xerr=sd,fmt='|',ms=10,color=INK,lw=1.1,capsize=3,zorder=2)
                else:ax.plot([mean,mean],[i-.25,i+.25],color=INK,lw=1.1)
            summary.append(dict(panel=letter,variant=v,scope=scope,metric=metric,n_runs=len(vals),mean=mean,sd=sd))
        ax.set(ylim=(3.65,-.6),yticks=range(4),yticklabels=[names[v] for v in variants] if letter in ['a','c'] else ['']*4,xlim=(.5,1.01) if metric=='auroc' else (0,1.02),xticks=[.5,.75,1] if metric=='auroc' else [0,.5,1],xlabel='AUROC' if metric=='auroc' else 'Rate at threshold 0.5');clean(ax);title(ax,letter,lab)
        ax.axhline(1.5,color='#C4CBD0',ls=':',lw=.8)
    fig.legend(handles=[Line2D([0],[0],marker='o',ls='',color=G,label='Initial seeds: 17, 29, 43'),Line2D([0],[0],marker='s',ls='',markerfacecolor='white',color=G,label='Added seeds: 59, 71'),Line2D([0],[0],marker='D',ls='',color=G,label='One logistic fit')],loc='upper center',bbox_to_anchor=(.52,.99),ncol=3,fontsize=6.6)
    fig.text(.02,.012,'Neural dots: individual seeds; black ticks: means; AUROC whiskers: sample SD, not confidence intervals.',fontsize=6.8)
    pd.DataFrame(summary).to_csv(S/'figure4_summary.csv',index=False)
    save(fig,'Figure_4_seed_stability_and_baselines')

def screening():
    d=pd.read_csv(A/'run_metrics_verified.csv');d=d[d.stage.eq('screen')&d.inference.eq('reference_bag')]
    d.to_csv(S/'figureS7_all_screen_seeds.csv',index=False)
    variants=list(names)[:10];fig=plt.figure(figsize=(7.2,5.0))
    for scope,letter,pos in [('internal','a',[.31,.16,.28,.71]),('external','b',[.68,.16,.28,.71])]:
        ax=fig.add_axes(pos)
        for i,v in enumerate(variants):
            z=d[d.variant.eq(v)&d.scope.eq(scope)].sort_values('seed');assert len(z)==3
            c=T if v=='attention_l2' else (B if v=='hybrid_control' else G)
            ax.errorbar(z.auroc.mean(),i,xerr=z.auroc.std(ddof=1),fmt='|',ms=10,capsize=2,color=INK,lw=.8)
            ax.scatter(z.auroc,np.array([-.16,0,.16])+i,color=c,s=20,zorder=3)
        ax.set(yticks=range(10),yticklabels=[names[v] for v in variants] if scope=='internal' else ['']*10,ylim=(9.6,-.6),xlim=(0,1.02),xticks=[0,.5,1],xlabel='AUROC');clean(ax);title(ax,letter,f'{scope.capitalize()} evaluation')
        ax.axhline(1.5,color=G,ls=':',lw=.8)
    fig.text(.02,.035,'Every initial run is shown (3 seeds per configuration). Whiskers: mean ± sample SD.\nTeal marks the internally selected candidate; row order follows the protocol, not external performance.',fontsize=7)
    save(fig,'Supplementary_Figure_S7_screening')

def aggregation():
    d=pd.read_csv(A/'ensemble_metrics.csv');d=d[~d['mode'].eq('additional_two_seed_reference')]
    d.to_csv(S/'figureS8_aggregation_metrics.csv',index=False)
    combos=[(v,'three_seed_reference') for v in list(names)[:10]]
    combos += [(v,mode) for mode in ['five_seed_reference','three_seed_five_bag','five_seed_five_bag'] for v in ['hybrid_control','attention_l2']]
    rowlabels=[names[v] for v,mode in combos]
    fig=plt.figure(figsize=(7.2,7.1))
    for scope,letter,left in [('internal','a',.38),('external','b',.71)]:
        ax=fig.add_axes([left,.14,.27,.74]);vals=[]
        for v,mode in combos:
            z=d[d.variant.eq(v)&d['mode'].eq(mode)&d.scope.eq(scope)];assert len(z)==1
            vals.append(z.iloc[0][['auroc','sensitivity','specificity']].to_numpy(float))
        matrix(ax,np.array(vals),['AUROC','Sens. (%)','Spec. (%)'],{1,2},rowlabels if scope=='internal' else None,fontsize=7)
        title(ax,letter,scope.capitalize())
        for y in [9.5,11.5,13.5]:ax.axhline(y,color='white',lw=4)
    for label,top,bottom in [('3 seeds, reference',-.5,9.5),('5 seeds, reference',9.5,11.5),('3 seeds, 5 bags',11.5,13.5),('5 seeds, 5 bags',13.5,15.5)]:
        y=.14+.74*(1-((top+bottom)/2+.5)/16)
        fig.text(.025,y,label,rotation=90,va='center',ha='center',fontsize=6.6,color=G)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap=CMAP),cax=fig.add_axes([.48,.067,.40,.017]),orientation='horizontal',ticks=[0,.5,1])
    cb.set_label('Cell color: metric value on a 0–1 scale',fontsize=6.7,labelpad=2)
    fig.text(.02,.012,'Metrics are calculated after probability averaging. All prespecified combinations are shown; no row replaces the primary checkpoint.',fontsize=6.6)
    save(fig,'Supplementary_Figure_S8_aggregation')

if __name__=='__main__':
    internal();external();stability();screening();aggregation()
    (S/'outputs.json').write_text(json.dumps({'figures':saved,'new_training':False,'new_inference':False},indent=2)+'\n')
    print('\n'.join(saved))

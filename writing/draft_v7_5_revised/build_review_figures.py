"""Render manuscript figures from archived outputs and completed supplementary analyses.

No training or model inference. Source tissue assets are displayed unchanged;
locators, scale bars and legends are separate vector annotations.
"""
from pathlib import Path
import csv,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/trident_v6_mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
from PIL import Image

R=Path(__file__).resolve().parent;D=R/'supporting_data';F=R/'figures';E=R/'experiments'/'outputs'
B='#336B91';T='#397D72';O='#B46B37';P='#77618E';G='#7B838B';INK='#263945'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.titlesize':8.5,
 'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.5,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'legend.frameon':False,'lines.linewidth':1.3,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
def data(name):return pd.read_csv(D/name)
def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(F/f'{name}.{ext}',dpi=450,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)
def letter(ax,s,x=-.12,y=1.1):ax.text(x,y,s,transform=ax.transAxes,weight='bold',fontsize=10,va='top')
def matrix(ax,row,title):
    z=np.array([[row['tn'],row['fp']],[row['fn'],row['tp']]],int)
    ax.imshow(z,cmap='Blues',vmin=0,vmax=z.max(),aspect='equal')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(z[i,j]),ha='center',va='center',fontsize=12,color='white' if z[i,j]>z.max()*.5 else INK)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['SS','MALT'],yticklabels=['SS','MALT'],xlabel='Predicted class',ylabel='Reference class',title=title)
    ax.tick_params(length=0)
def box(ax,x,y,w,h,text,color='#EAF1F5',size=7.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.3,rounding_size=1',lw=.8,ec='#7D96A6',fc=color))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=INK)
def arrow(ax,start,end,color=B,style='-|>',connectionstyle='arc3'):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle=style,mutation_scale=8,lw=.85,color=color,connectionstyle=connectionstyle))
def linearrow(ax,points,color=B):
    for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=.85)
    arrow(ax,points[-2],points[-1],color)

M=data('internal_model_metrics.csv');MD={r['aggregator']:r for r in M.to_dict('records')}
pred={f.stem.replace('internal_predictions_',''):pd.read_csv(f) for f in D.glob('internal_predictions_*.csv')}

def derive():
    cases=data('interpretation_cases.csv');main=pred['hybrid_evidence_mil'].set_index('study_slide_id')
    result=[]
    for x in cases.to_dict('records'):
        p=float(main.loc[x['study_slide_id'],'probability']);q=float(x['probability'])
        result.append({**x,'main_evaluation_probability':p,'interpretation_probability':q,
            'absolute_probability_difference':abs(p-q),'main_prediction_at_0_5':int(p>=.5),
            'interpretation_prediction_at_0_5':int(q>=.5),'classification_agrees':(p>=.5)==(q>=.5)})
    pd.DataFrame(result).to_csv(D/'interpretation_probability_comparison.csv',index=False)
    configs=[('attention_only',[1,0,0,0,0],0,'learned, 5 positions'),('attention_mean_std',[1,1,1,0,0],0,'learned, 5 positions'),
        ('attention_mean_std_proto',[1,1,1,1,0],1,'learned, 5 positions'),('full_equal_gate',[1,1,1,1,1],1,'equal'),('full_learned_gate',[1,1,1,1,1],1,'learned, 5 positions')]
    pd.DataFrame([dict(configuration=k,attention=v[0],mean=v[1],standard_deviation=v[2],positive_margin_vector=v[3],negative_plus_regional=v[4],eight_statistics=s,gate=g) for k,v,s,g in configs]).to_csv(D/'configuration_activation_matrix.csv',index=False)
    ext=data('external_final_predictions.csv');out=[]
    for (cohort,tag,label),g in ext.groupby(['cohort','stain_group','label']):
        out.append(dict(cohort=cohort,archive_stain_tag=tag,label=int(label),n=len(g),correct_at_0_5=int((g.label==g.prediction_at_0_5).sum()),median_probability=float(g.malt_probability.median())))
    pd.DataFrame(out).to_csv(D/'external_archive_tag_counts.csv',index=False)
    (D/'external_context_metrics.json').write_text(json.dumps({'n':168,'n_malt':152,'n_ss':16,'balanced_accuracy':(142/152+11/16)/2,'constant_positive_accuracy':152/168,'source':'Arithmetic summaries of unchanged external predictions; not a new evaluation'},indent=2)+'\n')

def fig1():
    fig=plt.figure(figsize=(7.2,7.3))
    ax=fig.add_axes([.025,.75,.95,.22]);ax.set(xlim=(-1,101),ylim=(-4,105));ax.axis('off');letter(ax,'a',-.035,1.10)
    ax.set_title('Study cohorts and evaluation',loc='left',pad=11)
    box(ax,0,34,18,40,'Internal cohort\n193 WSIs\nMALT 90 | SS 103',size=7)
    box(ax,25,61,21,37,'Training\n144 WSIs\nMALT 67 | SS 77',size=7)
    box(ax,25,5,21,37,'Internal evaluation\n49 WSIs\nMALT 23 | SS 26',size=7)
    box(ax,55,61,20,37,'Hybrid-Evidence\nMIL training',color='#DDECF2',size=7.2)
    box(ax,55,5,20,37,'Internal\nperformance',color='#DDECF2',size=7.2)
    box(ax,82,25,18,49,'External evaluation\n168 WSIs\nMALT 152\nSS 16',color='#F7EBDD',size=6.8)
    linearrow(ax,[(18,64),(21,64),(21,80),(25,80)],G);linearrow(ax,[(18,43),(21,43),(21,24),(25,24)],G)
    arrow(ax,(46,80),(55,80),G);arrow(ax,(46,24),(55,24),G);arrow(ax,(65,61),(65,42),B)
    linearrow(ax,[(75,79),(91,79),(91,74)],B)
    ax.text(68,49,'model',fontsize=6.5,color=B);ax.text(1,-3,'All counts are whole-slide images. Gray arrows: data; blue arrows: trained model.',fontsize=6.4,color=G)
    ax=fig.add_axes([.025,.045,.95,.64]);ax.set(xlim=(-3,104),ylim=(-5,103));ax.axis('off');letter(ax,'b',-.035,1.045)
    ax.set_title('Five evidence vectors with a separate statistics pathway',loc='left',pad=11)
    for x,txt in [(0,'Tissue segmentation\nGrandQC'),(26,'20× tissue patches\n512 × 512 pixels'),(52,'UNI2-h\n1,536-d embeddings'),(78,'Sample 512 patches\nfrom each slide')]:box(ax,x,89,22,11,txt,size=7)
    for x in [22,48,74]:arrow(ax,(x,94.5),(x+4,94.5))
    box(ax,20,77,60,7,'Feature projection → 512 vectors × 256 dimensions',size=7.5)
    linearrow(ax,[(89,89),(89,80.5),(80,80.5)])
    for x in [9,29,49]:linearrow(ax,[(50,77),(50,74),(x,74),(x,60)])
    box(ax,62,65,38,8,'Latent-bank similarities → margins\n8 prototypes in each of 2 banks',color='#E5F0E9',size=7)
    linearrow(ax,[(60,77),(60,69),(62,69)])
    for x in [69,90]:arrow(ax,(x,65),(x,60))
    labels=['Attention\ntransformed-value\nweighted pooling','Feature mean\nsampled-bag\ncenter','Feature SD\nsampled-bag\nvariation','High margin\ntop-32\nfeature mean','Low-margin mean\n+ regional pool\n≤16 ordered groups']
    for x,txt,col in zip([0,20,40,60,80],labels,['#E4EEF6','#E6F2EE','#E9EDF4','#E5F1E9','#F7EBDD']):box(ax,x,44,18,16,txt,col,7)
    for x in [9,29,49,69,89]:linearrow(ax,[(x,44),(x,42),(30,42),(30,40)])
    box(ax,1,32,58,8,'Concatenate the five unweighted vectors',size=7.2)
    box(ax,72,32,28,8,'8 similarity / margin\nstatistics',color='#F3EFDF',size=7)
    linearrow(ax,[(100,69),(102,69),(102,36),(100,36)],O)
    box(ax,35,20,30,8,'Softmax gate: five weights',color='#E7EFF6',size=7.3)
    linearrow(ax,[(44,32),(44,30),(42,30),(42,28)]);linearrow(ax,[(78,32),(78,24),(65,24)],O)
    box(ax,20,9,60,7,'Weighted vector concatenation + 8 statistics',color='#E2EDF3',size=7.3)
    arrow(ax,(50,20),(50,16));linearrow(ax,[(1,36),(-1,36),(-1,12.5),(20,12.5)])
    ax.text(0,22,'vectors',fontsize=6.2,color=B)
    linearrow(ax,[(94,32),(98,32),(98,12.5),(80,12.5)],O);ax.text(83,20,'statistics bypass',fontsize=6.1,color=O)
    box(ax,20,-2,60,7,'Fusion MLP → binary classifier → uncalibrated MALT score',color='#E6F0ED',size=7.3);arrow(ax,(50,9),(50,5))
    save(fig,'Figure_1_study_framework')

def fig2():
    fig,ax=plt.subplots(2,2,figsize=(7.2,6.2),gridspec_kw={'hspace':.72,'wspace':.65})
    selected=[('hybrid_evidence_mil',B,'Hybrid'),('abmil',G,'ABMIL'),('mean',P,'Mean pooling'),('hybrid_clam_rrt_ensemble',T,'Ensemble')]
    for k,col,name in selected:
        v=pred[k];m=MD[k];fpr,tpr,_=roc_curve(v.label,v.probability);prec,rec,_=precision_recall_curve(v.label,v.probability)
        ax[0,0].plot(fpr,tpr,color=col,label=f'{name} ({m["auroc"]:.3f})');ax[0,1].plot(rec,prec,color=col,label=f'{name} ({m["auprc"]:.3f})')
    for a in ax[0]:a.set(xlim=(0,1.01),ylim=(0,1.03));a.legend(loc='lower left',fontsize=6)
    ax[0,0].plot([0,1],[0,1],':',color=G,lw=.7);ax[0,0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='Internal ROC | 23 MALT / 26 SS')
    ax[0,1].set(xlabel='Recall',ylabel='Precision',title='Internal precision–recall')
    tab=M[M.aggregator!='titan'].sort_values('auroc');titan=M[M.aggregator=='titan'];tab=pd.concat([titan,tab],ignore_index=True)
    ys=np.r_[0,np.arange(1,len(tab))+.65]
    for j,row in tab.iterrows():
        col=O if row.aggregator=='titan' else B
        ax[1,0].errorbar(row.auroc,ys[j],xerr=[[row.auroc-row.auroc_ci_low],[row.auroc_ci_high-row.auroc]],fmt='o',color=col,ms=3.3,capsize=2,lw=.8)
    ax[1,0].axhline(.75,color=G,ls=':',lw=.8)
    ax[1,0].set(yticks=ys,yticklabels=tab.model,xlim=(.65,1.015),xlabel='AUROC (95% bootstrap interval)',title='Model configurations')
    ax[1,0].tick_params(axis='y',labelsize=5.8)
    matrix(ax[1,1],MD['hybrid_evidence_mil'],'Hybrid | threshold 0.5')
    for a,l in zip(ax.flat,'abcd'):letter(a,l)
    fig.text(.07,.005,'Hybrid − ABMIL: ΔAUROC 0.0100 (95% CI −0.0184 to 0.0452).\nOrange: different encoder and patch budget; blue: UNI2-h configurations.',fontsize=6.5,color=INK)
    save(fig,'Figure_2_internal_performance')

def fig3():
    e=data('external_final_predictions.csv');m=json.loads((D/'external_final_metrics.json').read_text())
    fig=plt.figure(figsize=(7.2,6.3));gs=fig.add_gridspec(2,2,height_ratios=[1,1.17],wspace=.48,hspace=.55)
    ref=pd.read_csv(E/'reference_predictions.csv');ref=ref[ref.scope=='external']
    axroc=fig.add_subplot(gs[0,0]);axpr=fig.add_subplot(gs[0,1])
    for model,col,name in [('hybrid_evidence_mil',B,'Hybrid'),('abmil',G,'ABMIL'),('clam_sb',P,'CLAM-style'),('rrtmil',T,'RRT-style')]:
        z=ref[ref.model==model];assert len(z)==168
        x,y,_=roc_curve(z.label,z.score);pr,rec,_=precision_recall_curve(z.label,z.score)
        axroc.plot(x,y,color=col,label=f'{name} ({roc_auc_score(z.label,z.score):.3f})')
        axpr.plot(rec,pr,color=col,label=f'{name} ({average_precision_score(z.label,z.score):.3f})')
    axroc.plot([0,1],[0,1],':',color=G)
    axpr.axhline(152/168,ls=':',color=G,lw=.8)
    axroc.set(xlim=(0,1),ylim=(0,1.03),xlabel='False-positive rate',ylabel='True-positive rate',title='External ROC | shared reference bags')
    axpr.set(xlim=(0,1),ylim=(0,1.03),xlabel='Recall',ylabel='Precision',title='External precision–recall')
    for a,l in [(axroc,'a'),(axpr,'b')]:a.legend(loc='lower right' if l=='a' else 'lower left',fontsize=6.0);letter(a,l)
    a=fig.add_subplot(gs[1,0]);matrix(a,m,'Hybrid | threshold 0.5');letter(a,'c')
    a=fig.add_subplot(gs[1,1]);groups=[('SS_BJ',None,'SS\nseries',P),('MALT_BJ','H&E-like','MALT\nH&E-like',O),('MALT_BJ','IHC / special stain','MALT\nIHC/special',B),('MALT_HE',None,'MALT\nHE archive',T)]
    rng=np.random.default_rng(14)
    for j,(cohort,tag,label,col) in enumerate(groups):
        z=e[e.cohort==cohort]
        if tag is not None:z=z[z.stain_group==tag]
        assert len(z)>0,(cohort,tag,e.stain_group.unique())
        a.scatter(j+rng.uniform(-.16,.16,len(z)),z.malt_probability,color=col,s=10,alpha=.65,linewidths=0)
        a.plot([j-.19,j+.19],[z.malt_probability.median()]*2,color=INK,lw=1.2)
        a.text(j,1.045,f'{int((z.label==z.prediction_at_0_5).sum())}/{len(z)}',ha='center',fontsize=6.3)
    a.axhline(.5,color=G,ls='--',lw=.7);a.set(xticks=range(4),xticklabels=[g[2] for g in groups],ylim=(-.02,1.16),ylabel='Uncalibrated MALT score',title='Archive tags | correct / total')
    a.tick_params(axis='x',labelsize=6.2);letter(a,'d')
    fig.text(.07,.005,'168 WSIs: 152 MALT / 16 SS. Archive tags await hospital-confirmed stain metadata.\nBalanced accuracy 81.09%; specificity 68.75% (95% CI 44.40–85.84%).',fontsize=6.5,color=INK)
    save(fig,'Figure_3_external_performance')

def fig4():
    readers=data('reader_aggregate_PROVISIONAL.csv');ens=MD['hybrid_clam_rrt_ensemble'];op=json.loads((D/'ensemble_threshold_0_70.json').read_text())
    fig=plt.figure(figsize=(7.2,5.9));gs=fig.add_gridspec(2,2,height_ratios=[1.2,.9],hspace=.65,wspace=.4)
    a=fig.add_subplot(gs[0,:]);x=np.arange(3);w=.18
    for j,(name,row,col) in enumerate([('Reader 1*',readers.iloc[0],G),('Reader 2*',readers.iloc[1],P),('Hybrid (0.5)',MD['hybrid_evidence_mil'],B),('Ensemble (0.5)',ens,T)]):
        bars=a.bar(x+(j-1.5)*w,[100*row[k] for k in ['accuracy','sensitivity','specificity']],width=w*.9,color=col,label=name,alpha=.88)
        if j<2:
            for bar in bars:bar.set_hatch('//')
    a.set(xticks=x,xticklabels=['Accuracy','Sensitivity','Specificity'],ylim=(0,105),ylabel='Percentage',title='Common model threshold: 0.5 | 49 WSIs')
    a.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.15),fontsize=6.5);letter(a,'a',-.045,1.12)
    a=fig.add_subplot(gs[1,0]);keys=['accuracy','sensitivity','specificity']
    for k in range(3):a.plot([100*ens[keys[k]],100*op[keys[k]]],[k,k],color='#C6CDD1',lw=1.2)
    a.scatter([100*ens[k] for k in keys],range(3),color=T,label='Ensemble 0.5',s=25)
    a.scatter([100*op[k] for k in keys],range(3),color=O,marker='s',label='Ensemble 0.70',s=25)
    a.set(yticks=range(3),yticklabels=['Accuracy','Sensitivity','Specificity'],xlim=(78,101),xlabel='Percentage',title='Separate operating point');a.invert_yaxis()
    a.legend(loc='upper center',bbox_to_anchor=(.5,-.24),fontsize=6.2,ncol=2);letter(a,'b')
    a=fig.add_subplot(gs[1,1]);a.axis('off');letter(a,'c',-.05,1.1)
    a.add_patch(FancyBboxPatch((.03,.10),.92,.78,boxstyle='round,pad=.025',fc='#FAF3E3',ec='#B2ACA1',lw=.7))
    a.text(.49,.49,'Paired reader analysis\n\nHospital-held individual records\npending incorporation',ha='center',va='center',fontsize=8)
    fig.text(.06,-.06,'* Provisional reader aggregates; reconcile with hospital-held individual records.\nNo AI-assisted benefit or reader-superiority inference is made.',fontsize=6.4,color='#775B26')
    save(fig,'Figure_4_reader_comparison')

def fig5():
    loc={x['publication_case']:x for x in json.loads((R/'source_data/roi_locations.json').read_text())}
    comp=data('interpretation_probability_comparison.csv').set_index('publication_case');gates=data('interpretation_gate_weights.csv')
    fig=plt.figure(figsize=(7.2,7.9));selected=[(1,'MALT | true positive'),(4,'SS | true negative'),(7,'SS | false positive'),(10,'MALT | false negative')]
    for row,(num,title) in enumerate(selected):
        y=.81-row*.18;v=comp.loc[f'Case {num}'];l=loc[f'Case {num}'];ax=fig.add_axes([.055,y,.375,.125]);image=np.asarray(Image.open(R/'assets'/f'case_{num:02d}_margin.png'))
        ax.imshow(image);ax.axis('off')
        fig.text(.025,y+.157,'abcd'[row],fontsize=10,weight='bold')
        fig.text(.075,y+.157,f'Case {num} · {title}',fontsize=7.7,weight='bold')
        fig.text(.075,y+.137,f'Score: main {v.main_evaluation_probability:.3f} | display {v.interpretation_probability:.3f}',fontsize=6.8)
        for j,(direction,col,lab) in enumerate([('positive','#B64E4F','H'),('negative','#416DA6','L')]):
            x0=l[direction+'_x']/l['level0_width']*image.shape[1];y0=l[direction+'_y']/l['level0_height']*image.shape[0]
            w=1024/l['level0_width']*image.shape[1];h=1024/l['level0_height']*image.shape[0]
            ax.add_patch(Rectangle((x0,y0),w,h,fill=False,ec='white',lw=2.3));ax.add_patch(Rectangle((x0,y0),w,h,fill=False,ec=col,lw=1.1))
            ax.annotate(lab,(x0+w/2,y0+h/2),xytext=(14 if j==0 else -14,-12 if j==0 else 14),textcoords='offset points',fontsize=7,color=col,weight='bold',bbox=dict(boxstyle='round,pad=.1',fc='white',ec=col,lw=.5),arrowprops=dict(arrowstyle='-',color=col,lw=.7))
            a=fig.add_axes([.505+j*.225,y,.18,.125]);a.imshow(np.asarray(Image.open(R/'assets'/f'case_{num:02d}_{direction}.png')));a.axis('off')
            a.set_title(f'{lab}: '+('Highest margin' if j==0 else 'Lowest margin'),fontsize=7,pad=2)
            width=50/l['patch_width_um']*220
            a.plot([12,12+width],[204,204],color='white',lw=4.2,solid_capstyle='butt');a.plot([12,12+width],[204,204],color='black',lw=2.0,solid_capstyle='butt')
            a.text(12,190,'50 µm',fontsize=5.8,color='black',bbox=dict(fc='white',ec='none',pad=.5))
    cax=fig.add_axes([.20,.225,.36,.012]);cb=fig.colorbar(ScalarMappable(norm=Normalize(-1,1),cmap='coolwarm'),cax=cax,orientation='horizontal',ticks=[-1,0,1])
    cb.ax.tick_params(labelsize=6);cb.set_label('Per-slide normalized prototype margin (m/s)',fontsize=6.6,labelpad=1)
    fig.text(.61,.226,'Blue: low margin\nRed: high margin (bank 1 − bank 0)',fontsize=6.5,va='center')
    ax=fig.add_axes([.24,.045,.65,.11]);chosen=[]
    for num,_ in selected:
        v=comp.loc[f'Case {num}'];r=gates[gates.study_slide_id==v.study_slide_id].iloc[0]
        cols=[f'gate_{j}' for j in range(1,6)]
        chosen.append([float(r[c]) for c in cols])
    a=np.array(chosen);assert a.shape==(4,5),a.shape
    im=ax.imshow(a,cmap='Blues',vmin=0,vmax=.4,aspect='auto')
    for i in range(4):
        for j in range(5):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=6.6,color='white' if a[i,j]>.27 else INK)
    ax.set(yticks=range(4),yticklabels=['Case 1: TP','Case 4: TN','Case 7: FP','Case 10: FN'],xticks=range(5),xticklabels=['Attention','Mean','SD','High\nmargin','Low margin +\nregional'],title='Gate weights (feature scaling coefficients)')
    ax.tick_params(length=0,labelsize=6.7);fig.text(.025,.168,'e',fontsize=10,weight='bold')
    ca=fig.add_axes([.91,.045,.012,.11]);fig.colorbar(im,cax=ca).set_label('Weight',fontsize=6.5)
    save(fig,'Figure_5_case_evidence')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),gridspec_kw={'wspace':.1})
    for ax,name,title,label in zip(axes,['malt','ss'],['MALT: illustrative photomicrograph','SS: illustrative photomicrograph'],'ab'):
        ax.imshow(np.asarray(Image.open(R/'assets'/f'illustrative_{name}.png')));ax.axis('off');ax.set_title(title,fontsize=8);letter(ax,label,-.015,1.1)
    fig.text(.07,.09,'Separate background examples; specimen linkage and original scale calibration remain pending.',fontsize=6.7,color='#775B26')
    save(fig,'Supplementary_Figure_S2_histological_context')

def fig_s3():
    ab=data('ablation_metrics.csv');seed=data('seed_runs.csv');cor=data('ensemble_member_correlations.csv')
    fig=plt.figure(figsize=(7.2,6.0));fig.text(.025,.97,'a',weight='bold',fontsize=10);fig.text(.07,.97,'Evidence configurations | batch 16, up to 30 epochs',fontsize=9)
    ax=fig.add_axes([.04,.56,.62,.35]);ax.axis('off')
    labels=['Attention vector','+ mean / SD','+ high-margin vector\n+ 8 statistics','Full: equal gate','Full: learned gate']
    flags=[['On','–','–','–','–','–','Learned'],['On','On','On','–','–','–','Learned'],['On','On','On','On','–','On','Learned'],['On','On','On','On','On','On','Equal'],['On','On','On','On','On','On','Learned']]
    tb=ax.table(cellText=[[lab]+row for lab,row in zip(labels,flags)],colLabels=['Configuration','Att.','Mean','SD','High','Low +\nreg.','8 stats','Gate'],cellLoc='center',colWidths=[.30,.075,.08,.065,.075,.10,.09,.135],bbox=[0,0,1,1])
    tb.auto_set_font_size(False);tb.set_fontsize(6.2)
    for (i,j),cell in tb.get_celld().items():
        cell.set_linewidth(.45);cell.set_edgecolor('#D4DDE3')
        if i==0:cell.set_facecolor('#E8EFF4');cell.set_text_props(weight='bold')
        elif j==0:cell.set_text_props(ha='left')
        elif cell.get_text().get_text()=='On':cell.set_facecolor('#E2EFEB')
    ax=fig.add_axes([.70,.56,.25,.35]);ys=np.arange(5)
    ax.scatter(ab.best_auc,ys,s=24,color=[G,G,G,T,B]);ax.set(yticks=[],ylim=(4.5,-1.5),xlim=(.915,.98),xlabel='AUROC')
    for i,v in enumerate(ab.best_auc):ax.text(v+.0015,i,f'{v:.4f}',va='center',fontsize=6.5)
    fig.text(.07,.505,'Inactive vectors are masked; learned gates still normalize over five positions.\nThe high-margin configuration also enables eight statistics. Main Hybrid uses batch 8.',fontsize=6.5,color=G)
    ax=fig.add_axes([.10,.12,.33,.27]);ax.plot(seed.seed.astype(str),seed.default_auc,'o',color=B,ms=5)
    for i,row in enumerate(seed.itertuples()):ax.text(i,row.default_auc+.010,f'{row.epochs} epochs',ha='center',fontsize=6.3)
    ax.set(ylim=(.82,1.0),xlabel='Seed',ylabel='AUROC',title='Different partitions / training settings');letter(ax,'b',-.22,1.19)
    ax=fig.add_axes([.62,.12,.28,.27]);members=['hybrid_evidence_mil','clam_sb','rrtmil']
    probs=pd.concat([pred[k].set_index('study_slide_id').probability.rename(k) for k in members],axis=1);a=probs.corr().values
    ax.imshow(a,cmap='Blues',vmin=.85,vmax=1)
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',fontsize=7,color='white' if a[i,j]>.94 else INK)
    ax.set(xticks=range(3),yticks=range(3),xticklabels=['Hybrid','CLAM-style','RRT-style'],yticklabels=['Hybrid','CLAM-style','RRT-style'],title='Ensemble score correlation');ax.tick_params(length=0,labelsize=6);letter(ax,'c',-.31,1.19)
    fig.text(.07,.015,'Configuration values and seed values represent individual runs, not matched replicate means.',fontsize=6.6,color=INK)
    save(fig,'Supplementary_Figure_S3_historical_configurations')

def fig_s1():
    d=pred['hybrid_evidence_mil']
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.1),gridspec_kw={'wspace':.5})
    rng=np.random.default_rng(421)
    for j,(lab,col,name) in enumerate([(0,P,'SS (n=26)'),(1,B,'MALT (n=23)')]):
        x=d[d.label==lab].probability.to_numpy()
        axs[0].scatter(j+rng.uniform(-.13,.13,len(x)),x,s=17,c=col,alpha=.8,linewidths=0)
        axs[0].plot([j-.18,j+.18],[np.median(x)]*2,c=INK,lw=1.1)
    axs[0].axhline(.5,color=G,ls='--',lw=.8)
    axs[0].set(xticks=[0,1],xticklabels=['SS (n=26)','MALT (n=23)'],ylabel='Uncalibrated MALT score',ylim=(-.02,1.02),title='Internal fixed-bag predictions')
    thresholds=np.linspace(0,1,201)
    sens=[float((d[d.label==1].probability>=t).mean()) for t in thresholds]
    spec=[float((d[d.label==0].probability<t).mean()) for t in thresholds]
    axs[1].plot(thresholds,sens,c=B,label='Sensitivity');axs[1].plot(thresholds,spec,c=P,label='Specificity')
    axs[1].axvline(.5,color=G,ls='--',lw=.8);axs[1].set(xlabel='Score threshold',ylabel='Proportion',ylim=(-.02,1.02),xlim=(0,1),title='Descriptive threshold sweep');axs[1].legend(loc='lower left')
    for ax,l in zip(axs,'ab'):letter(ax,l)
    save(fig,'Supplementary_Figure_S1_subgroups_thresholds')

def fig6():
    x=pd.read_csv(E/'repeated_bag_predictions.csv')
    assert len(x)==4340 and x.groupby(['scope','study_slide_id']).size().eq(20).all()
    slides=x.groupby(['scope','study_slide_id','label']).score.agg(['mean','std','min','max']).reset_index()
    slides['crosses_0_5']=(slides['min']<.5)&(slides['max']>=.5)
    scores=x.groupby(['scope','repeat_id']).apply(lambda z:roc_auc_score(z.label,z.score),include_groups=False).rename('auroc').reset_index()
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.0),gridspec_kw={'hspace':.67,'wspace':.52})
    rng=np.random.default_rng(88)
    for j,(scope,col,n) in enumerate([('internal',B,49),('external',O,168)]):
        z=slides[slides.scope==scope];assert len(z)==n
        axs[0,0].scatter(j+rng.uniform(-.17,.17,n),z['std'],s=11,alpha=.65,color=col,linewidths=0)
        axs[0,0].plot([j-.23,j+.23],[z['std'].median()]*2,color=INK,lw=1.4)
        axs[0,0].text(j,float(z['std'].max())+.008,f'median {z["std"].median():.3f}',ha='center',fontsize=6.5)
        v=scores[scores.scope==scope].auroc
        axs[0,1].scatter(j+rng.uniform(-.15,.15,len(v)),v,color=col,s=15,alpha=.8,linewidths=0)
        axs[0,1].plot([j-.23,j+.23],[v.median()]*2,color=INK,lw=1.4)
        original=MD['hybrid_evidence_mil']['auroc'] if scope=='internal' else json.loads((D/'external_final_metrics.json').read_text())['auroc']
        axs[0,1].scatter(j,original,color='white',edgecolors=INK,marker='D',s=32,zorder=5)
    axs[0,0].set(xticks=[0,1],xticklabels=['Internal\nn = 49 WSIs','External\nn = 168 WSIs'],ylabel='Within-slide score SD',title='Sensitivity to patch sampling',ylim=(0,float(slides['std'].max())+.035))
    axs[0,1].set(xticks=[0,1],xticklabels=['Internal','External'],ylabel='AUROC per complete-cohort draw',title='20 draws | one fixed checkpoint',ylim=(max(0,float(scores.auroc.min())-.025),min(1,float(scores.auroc.max())+.025)))
    axs[0,1].text(.03,.04,'◇ Historical reference bag\n— Median of 20 additional draws',transform=axs[0,1].transAxes,fontsize=6.1,color=G)
    a=axs[1,0]
    counts=[]
    for j,(scope,lab,col) in enumerate([('internal',0,B),('internal',1,B),('external',0,O),('external',1,O)]):
        z=slides[(slides.scope==scope)&(slides.label==lab)];k=int(z.crosses_0_5.sum());n=len(z)
        a.bar(j,100*k/n,color=col,width=.64,alpha=.83)
        a.text(j,100*k/n+1.4,f'{k}/{n}',ha='center',fontsize=7)
        counts.append(100*k/n)
    a.set(xticks=range(4),xticklabels=['Internal\nSS','Internal\nMALT','External\nSS','External\nMALT'],ylabel='WSIs crossing threshold 0.5 (%)',title='Different decisions across 20 bags',ylim=(0,max(30,max(counts)+8)))
    a.tick_params(axis='x',labelsize=6.4)
    h=pred['hybrid_evidence_mil'].set_index('study_slide_id');ens=pred['hybrid_clam_rrt_ensemble'].set_index('study_slide_id').loc[h.index]
    hc=(h.probability>=.5).to_numpy()==h.label.to_numpy();ec=(ens.probability>=.5).to_numpy()==h.label.to_numpy()
    tab=np.array([[int((hc&ec).sum()),int((hc&~ec).sum())],[int((~hc&ec).sum()),int((~hc&~ec).sum())]])
    assert tab.sum()==49
    a=axs[1,1];a.imshow(tab,cmap='Blues',vmin=0,vmax=42)
    for i in range(2):
        for j in range(2):a.text(j,i,str(tab[i,j]),ha='center',va='center',fontsize=14,color='white' if tab[i,j]>21 else INK)
    a.text(1,.25,'MALT missed',ha='center',fontsize=6.2,color=O)
    a.text(0,1.25,'SS corrected',ha='center',fontsize=6.2,color=T)
    a.set(xticks=[0,1],yticks=[0,1],xticklabels=['Correct','Incorrect'],yticklabels=['Correct','Incorrect'],xlabel='Ensemble at 0.5',ylabel='Hybrid at 0.5',title='Error exchange | 49 internal WSIs')
    a.tick_params(length=0)
    for a,l in zip(axs.flat,'abcd'):letter(a,l,x=-.38 if l=='d' else -.12)
    fig.text(.07,.006,'Sampling repeats are technical repeats, not additional WSIs or training runs.\nPanel d uses archived fixed-bag predictions; panels a–c use the 20 new bags.',fontsize=6.5,color=INK)
    save(fig,'Figure_6_sampling_and_errors')

def fig_s4():
    c=R/'experiments'/'controlled_training'
    assert json.loads((c/'runtime.json').read_text())['complete']
    runs=pd.read_csv(c/'run_metrics.csv');assert len(runs)==18
    variants=['matched_attention','fixed_scaling','learned_gate']
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.7),gridspec_kw={'wspace':.43})
    for ax,scope,letter_id in zip(axs,['internal','external'],'ab'):
        z=runs[runs.scope==scope]
        for seed,col,marker in [(17,B,'o'),(29,O,'s'),(43,T,'^')]:
            p=z[z.seed==seed].set_index('variant').loc[variants]
            ax.plot(range(3),p.auroc,marker=marker,color=col,lw=.95,ms=4,label=f'Seed {seed}',alpha=.85)
        mean=z.groupby('variant').auroc.mean().loc[variants]
        for j,m in enumerate(mean):ax.plot([j-.11,j+.11],[m,m],color=INK,lw=1.8)
        ax.set(xticks=range(3),xticklabels=['Matched\nattention','Fixed\nscaling','Learned\ngate'],ylabel='AUROC',title=f'{scope.capitalize()} | '+('49 WSIs' if scope=='internal' else '168 WSIs'))
        span=z.auroc.max()-z.auroc.min();pad=max(.015,span*.25)
        ax.set_ylim(max(0,z.auroc.min()-pad),min(1,z.auroc.max()+pad));letter(ax,letter_id)
        ax.legend(loc='lower left',fontsize=6.2)
    fig.text(.07,-.025,'Lines join the same training seed across configurations; black marks show three-seed means.\nFixed split, training bags, optimization and evaluation bags; three seeds are not independent WSIs.',fontsize=6.5,color=INK)
    save(fig,'Supplementary_Figure_S4_controlled_training')

if __name__=='__main__':
    derive()
    for fn in [fig1,fig2,fig3,fig4,fig5,fig6,fig_s1,fig_s3]:
        fn();print(fn.__name__+' complete',flush=True)
    controlled=R/'experiments'/'controlled_training'/'runtime.json'
    if controlled.exists() and json.loads(controlled.read_text()).get('complete'):
        fig_s4();print('fig_s4 complete',flush=True)

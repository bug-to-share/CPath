"""Update the reader-panel status while retaining all saved aggregate values."""
from pathlib import Path
import csv
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/trident-paper-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parent


def read(name):
    with (ROOT/'supporting_data'/name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':8,
        'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
        'legend.fontsize':6.2,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.65,'lines.linewidth':1.35,'svg.fonttype':'none','pdf.fonttype':42,
        'savefig.facecolor':'white'})
    readers=read('reader_aggregate_PROVISIONAL.csv')
    hybrid=next(r for r in read('internal_model_metrics.csv') if r['aggregator']=='hybrid_evidence_mil')
    ensemble=json.loads((ROOT/'supporting_data/ensemble_threshold_0_70.json').read_text())
    fig=plt.figure(figsize=(7.20,3.6));gs=fig.add_gridspec(1,2,width_ratios=[1.65,1],wspace=.38)
    ax=fig.add_subplot(gs[0]);x=np.arange(3);w=.19
    for j,(name,row,col) in enumerate([
        ('Reader 1*',readers[0],'#7B838B'),('Reader 2*',readers[1],'#77618E'),
        ('Hybrid (0.5)',hybrid,'#336B91'),('Ensemble (0.70)',ensemble,'#397D72')]):
        values=[100*float(row[k]) for k in ['accuracy','sensitivity','specificity']]
        bars=ax.bar(x+(j-1.5)*w,values,width=w*.92,color=col,label=name,alpha=.85)
        if j<2:
            for bar in bars:bar.set_hatch('//')
    ax.set(xticks=x,xticklabels=['Accuracy','Sensitivity','Specificity'],ylabel='Percentage',
           ylim=(0,106),title='Available aggregate comparison | n = 49')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,fontsize=6)
    ax.text(-.12,1.13,'a',transform=ax.transAxes,fontweight='bold',fontsize=9,va='top')
    ax=fig.add_subplot(gs[1]);ax.axis('off')
    ax.text(-.05,1.13,'b',transform=ax.transAxes,fontweight='bold',fontsize=9,va='top')
    ax.add_patch(FancyBboxPatch((.02,.17),.94,.67,boxstyle='round,pad=0.012',
                               facecolor='#FAF0D8',edgecolor='#87949E',lw=.65))
    ax.text(.49,.505,'HOSPITAL-HELD\nREADER RECORDS\n\nPaired case analysis\npending incorporation',
            ha='center',va='center',fontsize=7)
    fig.text(.07,.015,'* Provisional aggregates; reconcile with hospital-held case records.',
             fontsize=6,color='#6B4A19')
    for suffix in ['pdf','svg','png']:
        fig.savefig(ROOT/'figures'/f'Figure_4_reader_comparison_DRAFT.{suffix}',
                    dpi=450,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)


if __name__=='__main__':main()

"""Fill evidence-derived text and export an editable manuscript and supplement."""
from pathlib import Path
import json
import re
import sys
import textwrap
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE=Path(__file__).resolve().parent
DATA=HERE/'source_data'

def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows])

def pct(x):return f'{float(x)*100:.2f}%'
def val(x):return f'{float(x):.4f}'
def ci(r,k,p=False):
    f=pct if p else val
    return f'{f(r[k])} ({f(r[k+"_ci_low"])}–{f(r[k+"_ci_high"])})'

def tables(internal,ext):
    sections=[]
    rows=[['Internal training','MALT',67,'[MISSING C02]'],['Internal training','SS',23,'[MISSING C02]'],
        ['Internal training','SP',54,'[MISSING C02]'],['Internal holdout','MALT',23,'[MISSING C02]'],
        ['Internal holdout','SS',11,'[MISSING C02]'],['Internal holdout','SP',15,'[MISSING C02]'],
        ['External selected','MALT: source B',144,'[MISSING C02]'],['External selected','MALT: HE directory',8,'[MISSING C02]'],
        ['External selected','SS (non-MALT)',16,'[MISSING C02]']]
    sections+=['### Table 1 | WSI cohort composition',md_table(['Analysis set','Diagnostic/source group','WSIs','Verified patients'],rows),
        'Non-MALT combines SS and SP internally. The external negative series contains SS only. [MISSING C02: add age, sex, clinical features, collection dates and verified patient counts.]']
    rows=[]
    for r in internal.to_dict('records'):
        rows.append([r['model'],val(r['auroc']),val(r['auprc']),pct(r['accuracy']),pct(r['sensitivity']),pct(r['specificity']),val(r['f1'])])
    sections+=['### Table 2 | Internal performance at the common threshold of 0.5',md_table(['Model','AUROC','AUPRC','Accuracy','Sensitivity','Specificity','F1'],rows),
        'n = 49 WSIs (23 MALT and 26 non-MALT). All rows use threshold 0.5. All models except CONCH v1.5 + TITAN use UNI2-h features. Interval estimates and predictive values appear in Supplementary Table S2. The ensemble at 0.70 is a separate archived operating point, not an additional model.']
    rows=[]
    for label,r in [('Internal Hybrid',internal[internal.aggregator.eq('hybrid_evidence_mil')].iloc[0].to_dict()),('External Hybrid',ext)]:
        for k,name,p in [('auroc','AUROC',False),('auprc','AUPRC',False),('accuracy','Accuracy',True),('sensitivity','Sensitivity',True),('specificity','Specificity',True),('ppv','PPV',True),('npv','NPV',True),('f1','F1',False)]:
            rows.append([label,name,ci(r,k,p)])
    sections+=['### Table 3 | Internal and external Hybrid estimates with 95% intervals',md_table(['Cohort','Metric','Estimate (95% interval)'],rows),
        'AUROC, average precision and F1 intervals use 2,000 label-stratified WSI bootstrap samples. Proportion intervals use Wilson intervals. Both cohorts use threshold 0.5.']
    rows=[['Reader 1','Provisional legacy aggregate','Not applicable','77.55%','73.91%','80.77%'],
        ['Reader 2','Provisional legacy aggregate','Not applicable','83.67%','82.61%','84.62%'],
        ['Hybrid','Saved model predictions','0.5','87.76%','95.65%','80.77%'],
        ['Three-model ensemble','Saved model predictions','0.70','93.88%','91.30%','96.15%']]
    sections+=['### Table 4 | Draft reader comparison with evidence status',md_table(['Reader/model','Evidence status','Threshold','Accuracy','Sensitivity','Specificity'],rows),
        '[MISSING R01/R02: verify reader aggregates using individual records and complete paired comparisons.] The AI rows use different explicitly identified thresholds and are not evidence of AI-assisted reader improvement.']
    return '\n\n'.join(sections)

def references():
    refs=json.loads((DATA/'references_metadata.json').read_text())
    out=[];bib=[];aud=[]
    for r in refs:
        num=r['number'];doi=r['doi'];m=r.get('crossref')
        if m:
            authors=[]
            for a in m.get('author',[])[:3]:
                initials=''.join(x[0] for x in re.split(r'[\s-]+',a.get('given','')) if x)
                authors.append(a.get('family','')+' '+initials)
            if len(m.get('author',[]))>3:authors.append('et al.')
            title=re.sub('<[^>]*>','',m.get('title',[''])[0]);journal=m.get('container-title',[''])[0]
            date=m.get('published-print',m.get('published',{})).get('date-parts',[[None]])[0][0]
            volume=m.get('volume','');pages=m.get('page',m.get('article-number',''))
            ref=', '.join(authors)+'. '+title.rstrip('.')+'. '+journal+'. '+str(date)
            if volume:ref+=';'+volume
            if pages:ref+=':'+pages
            ref+='.'
        else:
            ref=r['original_reference'];ref=re.sub(r'\s*doi:\S+','',ref).rstrip('.')+'.'
            if r['original_number']==20:
                ref='Ilse M, Tomczak JM, Welling M. Attention-based deep multiple instance learning. Proceedings of Machine Learning Research. 2018;80:2127–2136.'
            if r['original_number']==22:
                ref='Shao Z, Bian H, Chen Y, et al. TransMIL: Transformer based correlated multiple instance learning for whole slide image classification. Advances in Neural Information Processing Systems. 2021;34:2136–2147.'
        ref=ref.replace('et al..','et al.')
        out.append(f'{num}. {ref} https://doi.org/{doi}')
        if m:
            bt=re.sub('<[^>]*>','',m['title'][0]);ba=' and '.join(a.get('family','')+', '+a.get('given','') for a in m.get('author',[]));by=str(date);bj=journal
        else:
            fallback={1:('Lymphomas complicating primary Sjögren’s syndrome: from autoimmunity to lymphoma','Nocturne, Gaetane and Pontarini, Elena and Bombardieri, Michele and Mariette, Xavier','2021','Rheumatology'),
                5:('Attention-based Deep Multiple Instance Learning','Ilse, Maximilian and Tomczak, Jakub and Welling, Max','2018','Proceedings of Machine Learning Research'),
                16:('TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification','Shao, Zhuchen and Bian, Hao and Chen, Yang and Wang, Yifeng and Zhang, Jian and Ji, Xiangyang and Zhang, Yongbing','2021','Advances in Neural Information Processing Systems')}
            bt,ba,by,bj=fallback[num]
        typ='inproceedings' if num in [5,14,15,16] else 'article'
        container='booktitle' if typ=='inproceedings' else 'journal'
        bib.append('@'+typ+'{ref'+str(num)+',\n  title={'+bt+'},\n  author={'+ba+'},\n  year={'+by+'},\n  '+container+'={'+bj+'},\n  doi={'+doi+'},\n  url={https://doi.org/'+doi+'}\n}')
        aud.append({'number':num,'doi':doi,'status':r['metadata_status']})
    (HERE/'references.bib').write_text('\n\n'.join(bib))
    pd.DataFrame(aud).to_csv(DATA/'reference_metadata_audit.csv',index=False)
    return '\n\n'.join(out)

def format_run(p,text):
    # Native editable text; missing-input markers receive persistent highlighting.
    pat=r'(\[(?:MISSING|AUTHOR CHECK)[^\]]*\]|\*\*[^*]+\*\*)'
    for part in re.split(pat,text):
        if not part:continue
        run=p.add_run(part.replace('**',''))
        if part.startswith('**'):run.bold=True
        if part.startswith('[MISSING') or part.startswith('[AUTHOR CHECK'):
            run.font.color.rgb=RGBColor.from_string('8B3C12')
            shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'FFF2CC');run._r.get_or_add_rPr().append(shade)

def setup(doc):
    sec=doc.sections[0];sec.page_width=Inches(8.27);sec.page_height=Inches(11.69)
    sec.top_margin=Inches(.75);sec.bottom_margin=Inches(.72);sec.left_margin=sec.right_margin=Inches(.75)
    sec.header_distance=sec.footer_distance=Inches(.3)
    normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(10.5)
    normal.paragraph_format.line_spacing=1.15;normal.paragraph_format.space_after=Pt(6)
    for name,size in [('Title',17),('Heading 1',13),('Heading 2',11),('Heading 3',10.5)]:
        s=doc.styles[name];s.font.name='Arial';s.font.size=Pt(size);s.font.color.rgb=RGBColor.from_string('253E50')
    h=sec.header.paragraphs[0];h.text='MALT versus non-MALT | Manuscript draft v1';h.style='Caption'
    f=sec.footer.paragraphs[0];f.alignment=2
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');f._p.append(field)

def export_docx(md,path):
    doc=Document();setup(doc);lines=md.splitlines();i=0;declarations=False
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line == '## Figure legends':
            # Each numbered figure starts its own page; do not strand a section title.
            i+=1;continue
        if line.startswith('| '):
            tablelines=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                tablelines.append(lines[i].strip());i+=1
            parsed=[[c.strip() for c in x.strip('|').split('|')] for x in tablelines]
            parsed=[r for r in parsed if not all(re.fullmatch(r':?-+:?',v) for v in r)]
            table=doc.add_table(rows=0,cols=len(parsed[0]));table.style='Light Shading Accent 1'
            for ri,row in enumerate(parsed):
                cells=table.add_row().cells
                for cell,txt in zip(cells,row):
                    cell.text='';p=cell.paragraphs[0];format_run(p,txt)
                    p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3)
                    for r in p.runs:r.font.size=Pt(8)
                    if ri==0:
                        for r in p.runs:r.bold=True
                trpr=table.rows[-1]._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');trpr.append(cant)
                if ri==0:
                    repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
            doc.add_paragraph();continue
        if line.startswith('!['):
            m=re.match(r'!\[(.*?)\]\((.*?)\)',line)
            if m:
                # Image follows its heading; preceding caption remains with it where possible.
                p=doc.add_paragraph();p.alignment=1
                width=5.95 if 'Figure_1_combined_final' in m.group(2) else 6.65
                p.add_run().add_picture(str(HERE/m.group(2)),width=Inches(width))
            i+=1;continue
        m=re.match(r'^(#{1,4}) (.*)',line)
        if m:
            n=len(m.group(1));title=m.group(2)
            if n==2 and title=='Data availability':declarations=True
            if n==2 and title=='Tables':declarations=False
            if n==3 and (title.startswith('Figure ') or title.startswith('Supplementary Figure') or title.startswith('Table 3 |')):doc.add_page_break()
            if n==2 and title in ['Tables','Figure legends','References']:doc.add_page_break()
            p=doc.add_paragraph(style='Title' if n==1 else 'Heading '+str(n-1));format_run(p,title)
            if declarations:
                p.paragraph_format.space_before=Pt(6);p.paragraph_format.space_after=Pt(2)
                for r in p.runs:r.font.size=Pt(10)
            i+=1;continue
        if re.match(r'^\d+\. ',line):
            p=doc.add_paragraph();format_run(p,line)
            p.paragraph_format.line_spacing=1.0;p.paragraph_format.space_after=Pt(3)
            for r in p.runs:r.font.size=Pt(9)
        elif line.startswith('- '):
            p=doc.add_paragraph(style='List Bullet');format_run(p,line[2:])
        else:
            p=doc.add_paragraph();format_run(p,line)
        if declarations:
            p.paragraph_format.line_spacing=1.0;p.paragraph_format.space_after=Pt(3)
            for r in p.runs:r.font.size=Pt(9.5)
        i+=1
    doc.save(path)

def supplement(internal,summary):
    out=['# Supplementary information','## Supplementary Table S1 | Archived model configurations']
    config=pd.read_csv(HERE.parents[1]/'scripts/result/ss_experiments/summary.csv')
    out.append(md_table(['Model','Seed','Epochs','LR','Batch','Patches','Best epoch'],[
        [r.aggregator,r.seed,r.epochs,r.lr,r.batch_size,r.num_patches,r.best_epoch] for r in config.itertuples()]))
    out+=['All rows refer to the internal MALT/non-MALT task. The ensemble has no separate training epochs.',
        '## Supplementary Table S2 | Complete internal interval estimates']
    for model in internal.model:
        r=internal[internal.model.eq(model)].iloc[0].to_dict()
        out+=['### '+model,md_table(['Metric','Estimate (95% interval)'],[[name,ci(r,k,p)] for k,name,p in [
            ('auroc','AUROC',False),('auprc','AUPRC',False),('accuracy','Accuracy',True),('sensitivity','Sensitivity',True),
            ('specificity','Specificity',True),('ppv','PPV',True),('npv','NPV',True),('f1','F1',False)]])]
    out+=['## Supplementary Table S3 | Separate archived ensemble operating point']
    r=summary['ensemble_at_0_70']
    out+=[md_table(['Metric','Estimate (95% interval)'],[[name,ci(r,k,p)] for k,name,p in [
        ('auroc','AUROC',False),('accuracy','Accuracy',True),('sensitivity','Sensitivity',True),('specificity','Specificity',True),('f1','F1',False)]]),
        'Threshold = 0.70; TN/FP/FN/TP = 25/1/2/21. This is an archived selected operating point.']
    out+=['## Supplementary Table S4 | Paired internal model comparisons']
    d=pd.read_csv(DATA/'paired_model_comparisons.csv')
    out.append(md_table(['Comparison','AUROC difference','95% interval','Discordant correct (Hybrid/other)','Exact McNemar P'],[
        [r.comparison,val(r.auroc_difference),f'{r.difference_ci_low:.4f} to {r.difference_ci_high:.4f}',f'{r.hybrid_only_correct}/{r.other_only_correct}',f'{r.mcnemar_exact_p:.3f}'] for r in d.itertuples()]))
    out+=['Comparisons use the same 49 WSI and threshold 0.5. No multiplicity adjustment was applied. Intervals use 2,000 paired label-stratified bootstrap samples.',
        '## Supplementary Table S5 | External selected subgroups']
    d=pd.read_csv(DATA/'external_subgroups.csv')
    out.append(md_table(['Source','Archived stain tag','Label','WSI','Correct at 0.5','Median probability'],[
        [r.cohort,r.stain_group,r.label,r.n,r.correct,f'{r.probability_median:.4f}'] for r in d.itertuples()]))
    out+=['The initial archive and the selected analysis are separate tables. Filename-derived case counts are not substituted for verified clinical patient counts.',
        '## Supplementary Table S6 | Interpretation cases']
    d=pd.read_csv(DATA/'interpretation_cases.csv',dtype={'slide_id':str})
    # Publication labels preserve private slide identities only in the local mapping CSV.
    d['publication_case']=['Case '+str(i+1) for i in range(len(d))]
    d.to_csv(DATA/'interpretation_case_publication_mapping.csv',index=False)
    out.append(md_table(['Publication label','Diagnostic group','Selection category','MALT probability'],[
        [r.publication_case,r.class_name,r.category,f'{r.probability:.4f}'] for r in d.itertuples()]))
    out+=['[MISSING P03: add pathologist-confirmed descriptions of each selected ROI.]',
        '## Supplementary Figure S1 | Internal subgroup predictions and threshold behavior',
        '**a,** Individual Hybrid probabilities for the 23 MALT, 11 SS and 15 SP holdout slides. The dashed line marks 0.5. **b,** Sensitivity and specificity across thresholds for the same fixed predictions. Threshold variation does not represent an additional validation experiment.',
        '![Supplementary Figure S1](figures/Supplementary_Figure_S1_subgroups_thresholds.png)']
    return '\n\n'.join(out)

def main():
    s=json.loads((DATA/'analysis_summary.json').read_text());internal=pd.read_csv(DATA/'internal_model_metrics.csv')
    h=internal[internal.aggregator.eq('hybrid_evidence_mil')].iloc[0].to_dict()
    en=internal[internal.aggregator.eq('hybrid_clam_rrt_ensemble')].iloc[0].to_dict();e=s['external']
    vals={}
    for prefix,r in [('H',h),('E',e),('ENS',en)]:
        vals.update({prefix+'_AUC':val(r['auroc']),prefix+'_AUC_CI':val(r['auroc'])+f' (95% CI {r["auroc_ci_low"]:.4f}–{r["auroc_ci_high"]:.4f})',
            prefix+'_AP':val(r['auprc']),prefix+'_ACC':pct(r['accuracy']),prefix+'_SENS':pct(r['sensitivity']),prefix+'_SPEC':pct(r['specificity']),
            prefix+'_PPV':pct(r['ppv']),prefix+'_NPV':pct(r['npv']),prefix+'_F1':val(r['f1'])})
    vals['ABMIL_AUC']=val(internal[internal.aggregator.eq('abmil')].auroc.iloc[0])
    corr=pd.read_csv(DATA/'ensemble_member_correlations.csv',index_col=0).values
    co=corr[np.triu_indices(3,1)];vals['CORR_RANGE']=f'{co.min():.3f}–{co.max():.3f}'
    for a,b in [('PYTHON','python'),('NUMPY','numpy'),('PANDAS','pandas'),('SCIPY','scipy'),('SKLEARN','sklearn'),('MPL','matplotlib')]:vals[a+'_VERSION']=s['environment'][b]
    vals['TABLES']=tables(internal,e);vals['REFERENCES']=references()
    md=(HERE/'manuscript_template.md').read_text()
    for k,v in vals.items():md=md.replace('{{'+k+'}}',v)
    # Keep administrative declarations with the prose, before stand-alone tables/figures.
    a=md.index('## Data availability');b=md.index('## References',a)
    declarations=md[a:b];md=md[:a]+md[b:]
    t=md.index('## Tables');md=md[:t]+declarations+'\n'+md[t:]
    assert not re.search(r'\{\{\w+\}\}',md),'Unfilled numeric template token'
    (HERE/'MALT_nonMALT_manuscript_v1.md').write_text(md)
    export_docx(md,HERE/'MALT_nonMALT_manuscript_v1.docx')
    sup=supplement(internal,s);(HERE/'Supplementary_information_v1.md').write_text(sup)
    export_docx(sup,HERE/'Supplementary_information_v1.docx')
    markers=[]
    for m in re.finditer(r'\[(MISSING|AUTHOR CHECK)\s+([^\]]+)\]',md):
        detail=m.group(2);key=re.match(r'([A-Z]\d+(?:/[A-Z]\d+)*)',detail)
        markers.append(dict(type=m.group(1),id=key.group(1) if key else detail,detail=detail))
    d=pd.DataFrame(markers).drop_duplicates();d.to_csv(DATA/'missing_information_register.csv',index=False)
    # Sanity checks prevent a polished artifact from silently changing the task or evidence.
    checks={'all_numeric_tokens_filled':'{{' not in md,'six_main_figures':len(re.findall(r'^### Figure [1-6] ',md,re.M))==6,
        'four_main_tables':len(re.findall(r'^### Table [1-4] ',md,re.M))==4,'task_non_malt_explicit':'SS and SP combined' in md,
        'missing_information_visible':len(markers)>0,'no_unsupported_first_claim':not bool(re.search(r'\b(first-ever|first study|unprecedented)\b',md,re.I)),
        'no_synthetic_reader_records':pd.read_csv(DATA/'reader_records_TEMPLATE_EMPTY.csv').empty}
    write={**checks,'body_word_count_approx':len(re.findall(r'\b[\w–-]+\b',md.split('## Tables')[0])),
        'missing_marker_occurrences':len(markers),'unique_marker_ids':sorted(d.id.unique().tolist())}
    (HERE/'document_checks.json').write_text(json.dumps(write,indent=2))
    assert all(checks.values()),checks
    print('Manuscript and supplementary DOCX/Markdown exported.',write)

if __name__=='__main__':main()

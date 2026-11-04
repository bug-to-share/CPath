"""Generate supplementary tables from completed analyses; no inference or fitting."""
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parent
E=R/'experiments'/'outputs'
A=R/'analysis'
names={'hybrid_evidence_mil':'Hybrid','abmil':'ABMIL','clam_sb':'CLAM-SB-style','rrtmil':'RRTMIL-style','hybrid_clam_rrt_ensemble':'Three-model ensemble'}
def fmt(x): return f'{x:.4f}'
def pct(x): return f'{100*x:.2f}\\%'
def row(xs): return ' & '.join(map(str,xs))+r' \\'+'\n'
def begin(title,label,cols):return '\\clearpage\n\\begin{table}[H]\\centering\n'+f'\\caption{{{title}}}\\label{{tab:{label}}}\n\\small\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n'
def end(note):return '\\bottomrule\n\\end{tabular}\n\\par\\medskip\\raggedright\\footnotesize '+note+'\n\\end{table}\n'
s=pd.read_csv(E/'sampling_summary.csv').set_index('scope')
out=begin('Sampling variability with the fixed Hybrid checkpoint','S10','p{.47\\textwidth}rr')
out+=row(['Summary','Internal','External'])+'\\midrule\n'
items=[('WSIs',lambda x:str(int(x.n))),('Additional bags per WSI',lambda x:str(int(x.n_draws))),('Median within-slide score SD',lambda x:fmt(x.score_sd_median)),('Within-slide score SD, interquartile range',lambda x:f'{x.score_sd_q25:.4f}--{x.score_sd_q75:.4f}'),('Maximum within-slide score SD',lambda x:fmt(x.score_sd_max)),('Median within-slide score range',lambda x:fmt(x.score_range_median)),('WSIs crossing 0.5 across the 20 bags',lambda x:f'{int(x.threshold_crossing_slides)}/{int(x.n)}'),('SS WSIs crossing 0.5',lambda x:f'{int(x.threshold_crossing_ss)}/{int(x.n_ss)}'),('MALT WSIs crossing 0.5',lambda x:f'{int(x.threshold_crossing_malt)}/{int(x.n_malt)}'),('Per-draw AUROC, mean (SD)',lambda x:f'{x.draw_auroc_mean:.4f} ({x.draw_auroc_sd:.4f})'),('Per-draw AUROC, range',lambda x:f'{x.draw_auroc_min:.4f}--{x.draw_auroc_max:.4f}'),('20-bag mean-score AUROC (secondary)',lambda x:fmt(x.mean_bag_auroc)),('20-bag mean-score sensitivity (secondary)',lambda x:pct(x.mean_bag_sensitivity)),('20-bag mean-score specificity (secondary)',lambda x:pct(x.mean_bag_specificity))]
for k,fn in items:out+=row([k,fn(s.loc['internal']),fn(s.loc['external'])])
out+=end('Each technical repeat uses 512 sampled feature rows and the same frozen checkpoint. Standard deviations use the sample definition. Crossing means that at least one score is below 0.5 and at least one is at or above 0.5 among the 20 additional bags. AUROC spread across draws is not a confidence interval. Mean-score results are secondary summaries across all 20 bags and do not replace the archived single-bag results. Source data: sampling\\_summary.csv, sampling\\_per\\_slide.csv and sampling\\_per\\_draw\\_metrics.csv.')
m=pd.read_csv(E/'reference_model_metrics.csv');m=m[m.scope=='external'];d=pd.read_csv(E/'external_paired_comparisons.csv')
out+=begin('External model comparisons on shared reference bags','S11','p{.18\\textwidth}p{.23\\textwidth}p{.23\\textwidth}p{.23\\textwidth}')
out+=row(['Model','AUROC (95\\% interval)','Sensitivity (95\\% interval)','Specificity (95\\% interval)'])+'\\midrule\n'
for v in m.itertuples():out+=row([names[v.model],f'{v.auroc:.4f} ({v.auroc_ci_low:.4f}--{v.auroc_ci_high:.4f})',f'{100*v.sensitivity:.2f}\\% ({100*v.sensitivity_ci_low:.2f}--{100*v.sensitivity_ci_high:.2f})',f'{100*v.specificity:.2f}\\% ({100*v.specificity_ci_low:.2f}--{100*v.specificity_ci_high:.2f})'])
out+='\\bottomrule\n\\end{tabular}\n\\medskip\n\\begin{tabular}{p{.32\\textwidth}rp{.27\\textwidth}}\n\\toprule\n'
out+=row(['Paired difference','AUROC difference','95\\% interval'])+'\\midrule\n'
for v in d.itertuples():out+=row([v.comparison.replace('abmil','ABMIL').replace('clam_sb','CLAM-SB-style').replace('rrtmil','RRTMIL-style'),fmt(v.auroc_difference),f'{v.difference_ci_low:.4f} to {v.difference_ci_high:.4f}'])
out+=end('All four original checkpoints were evaluated on identical per-slide reference bags from 168 WSIs (152 MALT and 16 SS). Sensitivity and specificity use threshold 0.5 and Wilson intervals. AUROC and paired-difference intervals in this new analysis use 10,000 label-stratified WSI bootstrap draws; the archived Hybrid intervals in Supplementary Table S21 use 2,000 draws. Differences are unadjusted descriptive comparisons of complete model implementations, not isolated aggregation mechanisms or confirmatory family-wise inference. Source data: reference\\_model\\_metrics.csv and external\\_paired\\_comparisons.csv.')
r=pd.read_csv(A/'roi_mask_coverage.csv')
out+=begin('Archived segmentation-contour coverage of the eight displayed regions','S12','llrr')
out+=row(['Case','Margin rank','Contour coverage','Pixel-match verified'])+'\\midrule\n'
for v in r.itertuples():out+=row([v.publication_case,'Highest' if v.margin_rank=='highest_margin' else 'Lowest',pct(v.mask_coverage_fraction),'Yes' if v.original_thumbnail_pixels_verified else 'No'])
out+=end('Coverage is the area of intersection between each 1,024\\,$\\times$\\,1,024 level-0-pixel window and the union of the archived segmentation polygons, divided by window area. Polygon holes were treated as tissue in the archived segmentation export. These values describe automated contour inclusion, not pathologist-confirmed tissue, lesion or artifact content. All eight original 220-pixel thumbnails were exactly reproduced from the current reader window using the archived resize rule; none was replaced. For Case 10, the historical overlay uses a 102,400\\,$\\times$\\,62,208 coordinate extent, whereas the current reader reports height 70,400; locators retain the historical overlay mapping. This extent discrepancy does not change the verified ROI windows or their physical scale. Source data: roi\\_mask\\_coverage.csv and roi\\_mask\\_provenance.json. Regional pathology interpretation remains pending hospital confirmation.')
b=pd.read_csv(A/'brier_descriptive.csv')
out+=begin('Descriptive Brier scores of archived predictions','S13','llrl')
out+=row(['Cohort','Model','WSIs','Brier score (95\\% interval)'])+'\\midrule\n'
for v in b.itertuples():out+=row([v.cohort.capitalize(),names[v.model],v.n,f'{v.brier_score:.4f} ({v.ci_low:.4f}--{v.ci_high:.4f})'])
out+=end('Brier score is the mean squared difference between the uncalibrated sigmoid score and the binary label (MALT = 1). Intervals use 10,000 label-stratified WSI bootstrap resamples of fixed predictions. Seeds are recorded in the source table. No calibration function was fitted to either evaluation cohort. Brier score combines several aspects of probabilistic prediction and does not independently isolate calibration or establish clinical utility. Source data: brier\\_descriptive.csv.')
c=pd.read_csv(A/'model_checkpoint_inventory.csv')
out+=begin('Dimensions and parameter counts of the evaluated complete models','S14','lrrrrr')
out+=row(['Model','Head input','Attention width','Parameters','Epoch budget','Batch'])+'\\midrule\n'
for v in c.itertuples():out+=row([names[v.model],v.classifier_input_dim,v.attention_head_dim,f'{v.parameter_count:,}',v.epochs_budget,v.batch_size])
out+=end('All models receive 1,536-dimensional UNI2-h features and use 512 sampled rows. Head input is the dimension passed from the slide encoder to the common classifier construction; its hidden width is 256. ABMIL retains 1,536-dimensional patch features with additional transformations before and after attention pooling, whereas Hybrid projects features to 256 dimensions and fuses five vectors with eight statistics. These differences, along with training budgets, prevent attribution to pooling alone. Parameter counts were obtained by strict loading of the archived checkpoints; they do not measure runtime, memory, or energy. Source data: model\\_checkpoint\\_inventory.csv and checkpoint\\_tensor\\_shapes.json.')
(R/'sections/new_analyses_tables.tex').write_text(out)
print('Generated supplementary Tables S10–S14')

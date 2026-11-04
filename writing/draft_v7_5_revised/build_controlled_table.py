"""Create the table for all nine predefined controlled training runs."""
from pathlib import Path
import json,pandas as pd
R=Path(__file__).resolve().parent
C=R/'experiments'/'controlled_training'
assert json.loads((C/'runtime.json').read_text())['complete']
x=pd.read_csv(C/'run_metrics.csv');assert len(x)==18
variants=['matched_attention','fixed_scaling','learned_gate']
names={'matched_attention':'Matched attention','fixed_scaling':'Fixed scaling','learned_gate':'Learned gate'}
out=r'''\clearpage
\begin{table}[H]\centering
\caption{All predefined controlled training runs}\label{tab:S15}
\small
\begin{tabular}{lrrrrrr}
\toprule
Configuration & Seed & Epoch & \shortstack{Internal\\AUROC} & \shortstack{External\\AUROC} & \shortstack{External\\sensitivity} & \shortstack{External\\specificity} \\
\midrule
'''
for variant in variants:
 for seed in [17,29,43]:
  z=x[(x.variant==variant)&(x.seed==seed)].set_index('scope');i=z.loc['internal'];e=z.loc['external']
  vals=[names[variant],str(seed),str(int(i.selected_epoch)),f'{i.auroc:.4f}',f'{e.auroc:.4f}',f'{100*e.sensitivity:.2f}\\%',f'{100*e.specificity:.2f}\\%']
  out+=' & '.join(vals)+r' \\'+'\n'
 out+='\\addlinespace\n'
counts={v:int(x[x.variant==v].parameters.iloc[0]) for v in variants}
out+=r'''\bottomrule
\end{tabular}
\par\medskip\raggedright\footnotesize Each run used the same original 144/49 training/evaluation partition, 30 epochs, batch size eight, learning rate $5\times10^{-5}$, weight decay $10^{-4}$, dropout 0.3 and positive-class loss weight $77/67$. Within each seed, configurations shared identical initial values for common tensors, training bags and minibatch order. The original internal and external reference bags were fixed. Epoch denotes the first epoch with the highest internal AUROC; external evaluation was performed after selection. All nine predefined runs are reported. Matched attention shares the 256-dimensional projection, attention block and classifier construction but omits the other vectors, statistics and fusion layer. Fixed scaling multiplies all five vectors by 0.2; learned gating also changes nonlinear capacity. Parameters: '''
out+=', '.join(f'{names[v].lower()} {counts[v]:,}' for v in variants)
out+=r'''. Thus these are controlled configuration comparisons, not parameter-matched or isolated single-component tests. Across-seed summaries describe training variation in the same cohort and do not add independent WSIs. No three-seed significance test is used. Source data: run\_metrics.csv, configuration\_summary.csv and paired\_seed\_differences.csv.
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth,height=.42\textheight,keepaspectratio]{figures/Supplementary_Figure_S4_controlled_training.pdf}
\caption{\textbf{Controlled configuration comparisons across three training seeds}. \textbf{a,b,} Internal and external AUROCs for every selected checkpoint from the nine predefined runs in Supplementary Table S15. Lines join the same seed across configurations and black horizontal marks denote three-seed means. Cohorts and evaluation inputs are unchanged across runs. Three seed values are descriptive training repeats rather than independent clinical observations; no confidence interval or inferential superiority claim is implied. Source data: run\_metrics.csv.}\label{fig:S4}
\end{figure}
'''
(R/'sections/controlled_training_table.tex').write_text(out)
p=R/'sections/supplement_content.tex';s=p.read_text();line='\\input{sections/controlled_training_table}\n'
if line not in s:s+='\n'+line
p.write_text(s)
print('Generated Table S15 and Figure S4 legend')

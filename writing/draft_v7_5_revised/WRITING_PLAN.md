# v7.5 writing plan — 4 November 2026

Purpose: revise the complete English research manuscript from v7; retain its original figures, tables and primary predictions. Generic clinical application paper; no target journal assumed.

Central claim: fixed UNI2-h features contain a retrospective slide-level signal for MALT–SS classification in the evaluated collections. The experiments do not establish an independent internal test, a stable benefit of architectural complexity, confirmed H&E-only external validation, or clinical utility.

Evidence and limits:
- Historical primary checkpoint: 49-slide internal validation AUROC 0.9599; those slides also selected epochs. External mixed-material set: 168 slides, AUROC 0.8721, 16 SS only.
- Original paired comparisons and nine controlled runs retained; local simplified implementations explicitly labeled.
- Completed bounded screen: 34 neural runs plus two deterministic logistic controls. Five-seed Hybrid 0.9348 internal / 0.8056 external versus normalized attention 0.9341 / 0.8061; target improvement was not achieved. Replicated controls are not additional independent experiments.
- Post hoc archive-tag sensitivity: 57 remaining slides; AUROC 0.8689, sensitivity 75.61%; not confirmed H&E-only validation.
- Technical sampling variation, image provenance and provisional reader counts retained. Clinical linkage/reference standards/reader records stay marked pending.

Paragraph map:
1. Introduction: clinical distinction → nearest disease studies → reusable representations and evaluation recipes → bounded question.
2. Results: material and selection status → historical discrimination → external composition and tag sensitivity → concise feature-based controls → provisional readers → regions/sampling.
3. Discussion: task signal → architecture evidence → external spectrum/operating point → development and patient-level limits → regional/reader evidence → next validation.
4. Main Methods: design, input, model, selection, comparator hierarchy, secondary analyses, statistics. Exact equations and implementation details move to Supplementary Methods.
5. Abstract drafted last; explicitly names selection reuse and mixed external material.

Order of drafting: Results, Introduction, Discussion/Conclusion, Methods, Abstract/title. No new training or selection based on external outcomes. Original Figure 1–6, Supplementary Figure S1–S4 and Table 1–4/S1–S15 retained; captions clarify validation semantics. New tables follow S15. Build both PDFs, verify source hashes and metrics, inspect rendering and package sources.

Layout update: invoke main tables and figures beside Results paragraphs, preserving their content/numbering and checking all 10 occur before Discussion.

Historical visualization stage (superseded): retain cohort Table 1; move the exact remaining main-table rows into S20–S22; replace the two main performance displays with interval/metric-profile plots, add a five-seed/simple-baseline main figure, retain original ROC/PR plots as S5/S6, and add screening/aggregation displays S7/S8. At that stage the main figure count was seven; all appear before Discussion. Preserve every original numerical result and source figure asset.

Current author clarification: keep numerical tables, use bold for best and underline for second-best distinct values within comparable groups, and share marks across ties. Restore four main tables and six figures. Main argument: Hybrid combines fixed UNI2-h features to distinguish the recorded MALT/SS diagnostic groups; internally selected primary checkpoint AUROCs are 0.9599 and 0.8721, with clinical interpretation bounded by validation reuse and cohort metadata. Retain the original introduction funnel and terminology. Keep the main prose focused on the selected checkpoint and ensemble; move full repeated-training Results and random-state details to the supplement. Do not make seed stability a main-paper claim or substitute external maxima for paired checkpoint results. Preserve all numerical records and the approved Figure 1b.

# Public export scope — 2026-11-04

This is a curated public export of the local `writing/` workspace. It follows the repository policy: publish research text, code and aggregate experimental results; retain patient/slide-level material locally. It is not a complete submission package or a release of clinical data.

## Included

- v7.5 and v8 manuscript LaTeX, bibliographies, screened schematic/aggregate figures, and rebuilt public reading PDFs.
- Aggregate model, cohort, configuration and training-run metrics; literature and manuscript reviews; revision plans and action lists.
- Analysis/training source code without embedded original identifiers. Scripts that read local clinical outputs require separately authorized inputs.

## Excluded

- Patient/slide mappings, clinical confirmation forms, original slide identifiers, per-slide predictions, per-slide feature/ROI metadata and linked case tables, even when publication-coded.
- Whole-slide images, tissue crops, case heatmaps, illustrative histology, individual-slide observation plots and corresponding binary figure assets.
- Model weights, feature arrays, sample-index arrays, local logs/build caches/QA screenshots, old source-data/code ZIPs, private documents, downloaded third-party full-text PDFs and private conversation links.
- Historical v1–v7 draft packages and snapshots outside this curated release. Historical review reports are retained as text.

## Visible changes in the public manuscript copies

Each main and supplementary PDF begins with an export notice. Original figure numbering and captions are retained, and withheld figures are replaced by explanatory boxes. Their captions describe the complete local archive; the boxes do not claim that the original figure is publicly available.

| Version | Omitted visual/table content |
|---|---|
| v7.5 | Main Figures 3, 5, 6; Supplementary Figures S1, S2; Supplementary Tables S6, S12. The unused archived visual-supplement definition also omits the individual-slide Figure 3 asset. |
| v8 | Main Figure 3; Supplementary Figures S4, S5; Supplementary Table S9. |

These omissions remove tissue/case material and individual-slide plots. Aggregate numerical results and model rankings are unchanged. Source Data availability statements and references in the original prose refer to the full local archive: **only the files listed in `PUBLIC_FILE_INDEX.csv` are part of this public export**. Reports refer to the original local manuscript page numbers, which can differ from the filtered PDFs.

The review synthesis retains reviewer text and the revision plan; local machine paths and private conversation links have been removed or replaced by repository links. These are model-generated critiques, not journal decisions.

## Build and validation

Run `bash build.sh` inside either draft directory with TeX Live/latexmk/BibTeX installed. The filtered LaTeX sources and public figures are sufficient for these reading copies. Full image regeneration, prediction recomputation and training are not possible from this public export alone; local scripts may expect data that is deliberately withheld. No experiment was rerun and no performance estimate was reselected for publication.

Before upload, file types, CSV headers/contents and text extracted from PDFs were screened against original identifiers from the local private mapping. Screened graphics were visually inspected. Public PDFs were rebuilt and checked for unresolved references and missing files. `PUBLIC_FILE_INDEX.csv` records the SHA-256 of each public file (excluding itself); the private identifier list and excluded filenames are not distributed.

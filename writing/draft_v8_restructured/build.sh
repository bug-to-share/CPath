#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p build
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build supplement.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build review.tex
cp build/main.pdf MALT_SS_manuscript_v8.pdf
cp build/supplement.pdf Supplementary_information_v8.pdf
cp build/review.pdf MALT_SS_manuscript_v8_with_line_numbers.pdf

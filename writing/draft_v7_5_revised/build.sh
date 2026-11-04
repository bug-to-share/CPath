#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
latexmk -pdf main.tex
latexmk -pdf supplement.tex
latexmk -pdf main_review.tex
cp build/main.pdf MALT_SS_manuscript_v7_5.pdf
cp build/supplement.pdf Supplementary_information_v7_5.pdf
cp build/main_review.pdf MALT_SS_manuscript_v7_5_with_line_numbers.pdf
printf 'Built v7.5 manuscript and supplementary information. Logs: build/.\n'

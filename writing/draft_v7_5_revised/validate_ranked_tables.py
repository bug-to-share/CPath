"""Check numerical preservation, descriptive ranks and current manuscript framing."""
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parent
Q = ROOT / 'qa'


def numbers(text):
    data = [line for line in text.splitlines()
            if '&' in line and re.search(r'\\\\\*?\s*$', line)]
    return re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?', '\n'.join(data))


baseline = json.loads((Q / 'ranked_table_baseline_cells.json').read_text())
for name, expected in baseline.items():
    assert numbers((ROOT / name).read_text()) == expected, name

hashes = json.loads((Q / 'ranked_tables_before.json').read_text())
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
           for name, digest in hashes.items())

# Parse rendered LaTeX data independently of the formatter implementation.
groups = defaultdict(list)
for name in baseline:
    label, block = None, -1
    for line in (ROOT / name).read_text().splitlines():
        found = re.search(r'\\label\{tab:([^}]+)\}', line)
        if found:
            label, block = found[1], -1
        if re.search(r'\\begin\{tabularx?\}', line):
            block += 1
        if '&' not in line or not re.search(r'\\\\\*?\s*$', line):
            continue
        cells = [x.strip() for x in line.split('&')]
        columns, group, ascending = [], '', False
        if label == '2': columns = list(range(1, 7))
        elif label == '4': columns = [3, 4, 5]
        elif label == 'S2': columns, group = [2], cells[1]
        elif label == 'S9': columns = [4]
        elif label == 'S11' and block == 0: columns = [1, 2, 3]
        elif label == 'S13': columns, group, ascending = [3], cells[0], True
        elif label == 'S15': columns, group = [3, 4, 5, 6], cells[1]
        elif label == 'S16': columns = [1, 2]
        elif label == 'S17': columns, group = [2, 3, 4, 5], (block, cells[1])
        elif label == 'S19': columns, group = [3, 4, 5], (cells[1], cells[2])
        for i, cell in enumerate(cells):
            found = re.match(r'^\$?(?:\\(textbf|mathbf|underline)\{(-?\d+(?:\.\d+)?)(?:\\%)?\}|(-?\d+(?:\.\d+)?))', cell)
            if found is None:
                continue
            mark, value = found[1], Decimal(found[2] or found[3])
            if i in columns:
                groups[label, str(group), i, ascending].append((value, mark))
            else:
                assert mark is None, (label, i, 'non-performance cell marked')

marks = 0
for group, entries in groups.items():
    order = sorted({v for v, _ in entries}, reverse=not group[-1])
    for value, mark in entries:
        rank = order.index(value)
        expected = None if len(entries) < 2 or rank > 1 else ('bold' if rank == 0 else 'underline')
        actual = 'bold' if mark in ('textbf', 'mathbf') else mark
        assert actual == expected, (group, value, actual, expected)
        marks += actual is not None

main_sections = ['frontmatter', 'introduction', 'results', 'discussion',
                 'conclusion', 'methods', 'declarations', 'figures', 'tables']
main_text = '\n'.join((ROOT / 'sections' / f'{name}.tex').read_text() for name in main_sections)
assert not re.search(r'\bseeds?\b', main_text, re.I)
assert 'TrainingStabilityFigure' not in main_text
assert all(x not in main_text for x in ('S20', 'S21', 'S22', 'fig:stability'))
assert 'highest internal validation AUROC selected epoch 15' in main_text
assert '0.9599' in main_text and '0.8721' in main_text
supp = (ROOT / 'supplement.tex').read_text()
assert 'sections/secondary_development_results' in supp
assert 'sections/relocated_numeric_tables' not in supp
assert 'sections/visual_supplement' not in supp
layout = json.loads(subprocess.check_output(['python', str(ROOT / 'validate_pdf_layout.py')], text=True))
report = dict(status='PASS', unchanged_data_and_asset_files=len(hashes),
              numeric_tables_preserved=list(baseline), ranked_groups=len(groups),
              correct_best_second_marks=marks, tied_display_values_ranked_equally=True,
              ranking_direction_checked=True, no_seed_emphasis_in_main=True,
              primary_checkpoint_and_paired_cohort_results_retained=True, pdfs=layout)
(Q / 'ranked_tables_check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

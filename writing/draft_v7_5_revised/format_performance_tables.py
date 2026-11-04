"""Apply descriptive best/runner-up marks without changing any numerical value.

Rank distinct displayed point estimates, sharing marks across ties. Confidence
limits, thresholds, sample counts and single-model cohort contrasts are not ranked.
"""
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent
FILES = ['tables', 'supplement_content', 'new_analyses_tables',
         'controlled_training_table', 'development_tables']
NUMBER = r'-?\d+(?:\.\d+)?(?:\\%)?'
MARK = re.compile(r'\\(textbf|mathbf|underline)\{(' + NUMBER + r')\}')
POINT = re.compile(r'^(\s*\$?)(' + NUMBER + r')')


def comparison(label, block, cells):
    """Return metric columns, comparable-row group, and metric direction."""
    if label == '2':
        return range(1, 7), 'all models', True
    if label == '4':
        return range(3, 6), 'displayed readers/models', True
    if label == 'S2':
        return [2], cells[1].strip(), True
    if label == 'S9':
        return [4], 'historical configurations', True
    if label == 'S11' and block == 0:
        return range(1, 4), 'external models', True
    if label == 'S13':
        return [3], cells[0].strip(), False
    if label == 'S15':
        return range(3, 7), 'matched seed ' + cells[1].strip(), True
    if label == 'S16':
        return [1, 2], 'all configurations', True
    if label == 'S17':
        return range(2, 6), f'panel {block}; ' + cells[1].strip(), True
    if label == 'S19':
        return range(3, 6), cells[1].strip() + '; ' + cells[2].strip(), True
    return [], '', True


def format_text(text, filename):
    lines = text.splitlines(keepends=True)
    groups = defaultdict(list)
    rows = {}
    label = None
    block = -1
    for n, line in enumerate(lines):
        hit = re.search(r'\\label\{tab:([^}]+)\}', line)
        if hit:
            label, block = hit[1], -1
        if re.search(r'\\begin\{tabularx?\}', line):
            block += 1
        if '&' not in line or label is None:
            continue
        cells = line.split('&')
        cols, context, higher = comparison(label, block, cells)
        for col in cols:
            if col >= len(cells):
                continue
            # Idempotent: strip only existing numerical marks, not row labels.
            cells[col] = MARK.sub(lambda m: m[2], cells[col])
            hit = POINT.match(cells[col])
            if not hit:
                continue
            value = Decimal(hit[2].replace(r'\%', ''))
            groups[label, context, col, higher].append((n, col, value))
            rows[n] = cells
    records = []
    for (label, context, col, higher), entries in groups.items():
        # A single estimate has no comparator, even if it is numerically large.
        if len(entries) < 2:
            continue
        ordered = sorted({v for _, _, v in entries}, reverse=higher)
        for n, col, value in entries:
            rank = ordered.index(value) + 1
            if rank > 2:
                continue
            cell = rows[n][col]
            hit = POINT.match(cell)
            command = ('mathbf' if '$' in hit[1] else 'textbf') if rank == 1 else 'underline'
            rows[n][col] = cell[:hit.start(2)] + '\\' + command + '{' + hit[2] + '}' + cell[hit.end(2):]
            records.append(dict(file=filename, table=label, group=context,
                                line=n + 1, column=col, value=str(value), rank=rank))
    for n, cells in rows.items():
        lines[n] = '&'.join(cells)
    return ''.join(lines), records


def format_file(path):
    text, records = format_text(path.read_text(), str(path.relative_to(ROOT)))
    path.write_text(text)
    return records


def main():
    records = []
    for name in FILES:
        records.extend(format_file(ROOT / 'sections' / f'{name}.tex'))
    (ROOT / 'qa' / 'ranked_table_marks.json').write_text(json.dumps(records, indent=2) + '\n')
    print(f'Applied {len(records)} best/second-best marks; all values retained.')


if __name__ == '__main__':
    main()

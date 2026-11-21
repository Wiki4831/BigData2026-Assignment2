"""Render Question 1 from the official schema and current dataset build record.

Usage: python scripts/draw_schema.py
Requires the system Graphviz `dot` command; no Python graphviz package is needed.
"""
import argparse
import csv
from datetime import datetime, timedelta
from html import escape
import json
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo

from preprocess import TABLES

ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    'Users': '#2463a0', 'Repositories': '#a35f15', 'PullRequests': '#7350a1',
    'Issues': '#287d54', 'IssueComments': '#39747e', 'PRReviews': '#b34e52',
    'PRReviewComments': '#337f85', 'Pushes': '#555e70',
}


def label(name, spec):
    rows = [f'<TR><TD COLSPAN="2" BGCOLOR="{COLORS[name]}">'
            f'<FONT COLOR="white"><B>{name}</B></FONT></TD></TR>']
    fields = [(spec['pk'], 'PK', '#e8f0fb')]
    fields += [(column, 'FK', '#fff5df') for column in spec['fks']]
    for column, kind, background in fields:
        rows.append(f'<TR><TD BGCOLOR="{background}"><B>{kind}</B></TD>'
                    f'<TD PORT="{column}" ALIGN="LEFT" BGCOLOR="{background}">'
                    f'{escape(column)}</TD></TR>')
    return ('<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" '
            'CELLPADDING="7" COLOR="#718096">' + ''.join(rows) + '</TABLE>>')


def make_dot():
    lines = [
        'digraph Assignment2 {',
        '  graph [rankdir=LR, bgcolor="white", pad="0.3", nodesep="0.7",',
        '         ranksep="1.15", splines=spline, outputorder=edgesfirst,',
        '         fontname="DejaVu Sans", fontsize=19, labelloc=t,',
        '         label="Assignment 2: Schema Graph\\n8 tables | 20 foreign keys | arrows: FK -> PK\\n",',
        '         dpi=160];',
        '  node [shape=plain, fontname="DejaVu Sans", fontsize=12];',
        '  edge [penwidth=1.4, arrowsize=0.8];',
        '  { rank=same; PRReviewComments; IssueComments; Pushes; }',
        '  { rank=same; PRReviews; Issues; }',
        '  { rank=same; Users; Repositories; }',
    ]
    for name, spec in TABLES.items():
        lines.append(f'  {name} [label={label(name, spec)}];')
    edges = []
    for name, spec in TABLES.items():
        for column, target in spec['fks'].items():
            pk = TABLES[target]['pk']
            record = {'source_table': name, 'foreign_key': column,
                      'target_table': target, 'primary_key': pk}
            edges.append(record)
            ports = 'e', 'w'
            if name == target:
                ports = 'n', 'n'
            tooltip = f'{name}.{column} -> {target}.{pk}'
            lines.append(f'  {name}:{column}:{ports[0]} -> {target}:{pk}:{ports[1]} '
                         f'[color="{COLORS[target]}", tooltip="{tooltip}"];')
    lines.append('}')
    return '\n'.join(lines) + '\n', edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/question1')
    args = parser.parse_args()
    if shutil.which('dot') is None:
        parser.error('Graphviz dot is required. Install Graphviz before running this script.')
    args.out.mkdir(parents=True, exist_ok=True)
    dot, edges = make_dot()
    assert len(TABLES) == 8 and len(edges) == 20
    source = args.out / 'schema.dot'
    source.write_text(dot)
    for extension in ('svg', 'png', 'pdf'):
        subprocess.run(['dot', f'-T{extension}', str(source), '-o',
                        str(args.out / f'schema.{extension}')], check=True)
    with (args.out / 'foreign_keys.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(edges[0]))
        writer.writeheader()
        writer.writerows(edges)

    build = json.loads((ROOT / 'data/build_summary.json').read_text())
    start = datetime.fromisoformat(build['period_utc_inclusive'])
    end = datetime.fromisoformat(build['period_utc_exclusive'])
    days = (end - start).days
    start_tw = start.astimezone(ZoneInfo('Asia/Taipei'))
    end_tw = end.astimezone(ZoneInfo('Asia/Taipei'))
    last_date = (end - timedelta(microseconds=1)).date()
    hours = int((end - start).total_seconds() // 3600)
    assert hours == build['downloaded_hours']
    text = f'''# Question 1: Schema Graph and Dataset Time Period

## Schema graph

The schema graph contains all **8 tables**, with **8 primary keys (PK)** and
**20 foreign keys (FK)**. Each directed edge runs from a foreign-key column to
the referenced primary-key column (**FK → PK**). Key columns are shown in each node.

![Schema graph: all 8 tables and all 20 foreign-key edges](schema.png)

The self-referencing edge is
`PRReviewComments.in_reply_to_id → PRReviewComments.comment_id`.
Separate foreign keys to the same table are drawn as separate edges, including
`PullRequests.author_id` and `PullRequests.merged_by_id`, and
`Issues.author_id` and `Issues.assignee_id`, all referencing `Users.user_id`.
Anonymized code columns such as `github_user_id`, `org_id`, and `head_repo_id`
are not foreign keys in the official schema and are not drawn as FK edges.

## Dataset time period

The dataset selected for model development covers **{start:%Y-%m-%d} through
{last_date:%Y-%m-%d}, inclusive, in UTC**. We downloaded **{hours:,} hourly
GH Archive files** spanning **{days} complete UTC days**, then ran the provided
`scripts/preprocess.py` to generate the eight anonymized tables.

| Time zone | Start (inclusive) | End (exclusive) |
|---|---|---|
| UTC | {start:%Y-%m-%d %H:%M:%S} | {end:%Y-%m-%d %H:%M:%S} |
| Taiwan (UTC+8) | {start_tw:%Y-%m-%d %H:%M:%S} | {end_tw:%Y-%m-%d %H:%M:%S} |

The interval above is the raw-data collection period. For a prediction at time
`t`, only rows with **row time < t** are eligible: `first_seen_at` for Users,
Repositories, PullRequests, Issues, and PRReviews; `created_at` for
IssueComments, PRReviewComments, and Pushes.

Sources: [official schema](../../SCHEMA.md), [assignment instructions](../../README.md),
and [local dataset build record](../../data/build_summary.json).
The graph was rendered with [Graphviz](https://graphviz.org/).

Graph exports: [SVG](schema.svg), [PDF](schema.pdf), [DOT source](schema.dot).
The complete [foreign-key list](foreign_keys.csv) contains one row per edge.
'''
    (args.out / 'question1.md').write_text(text)
    print(f'Created {args.out}: 8 tables, 8 PKs, 20 FK edges, {hours} hourly files.')


if __name__ == '__main__':
    main()

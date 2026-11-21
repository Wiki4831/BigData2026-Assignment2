"""Create a native, editable draw.io schema and matching initial-layout previews.

Run: python scripts/draw_schema_drawio.py
The .drawio master has editable table containers, key rows and attached FK edges.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as PlotPath

from preprocess import TABLES

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1960, 1440
TABLE_WIDTH, HEADER, ROW = 340, 40, 36
COLORS = {
    'Users': '#2463a0', 'Repositories': '#a35f15', 'PullRequests': '#7350a1',
    'Issues': '#287d54', 'IssueComments': '#39747e', 'PRReviews': '#b34e52',
    'PRReviewComments': '#337f85', 'Pushes': '#555e70',
}
POSITIONS = {
    'PRReviewComments': (60, 300), 'IssueComments': (60, 770), 'Pushes': (60, 1100),
    'PRReviews': (560, 340), 'Issues': (560, 800), 'PullRequests': (1060, 600),
    'Users': (1560, 450), 'Repositories': (1560, 950),
}
# Intermediate waypoints; endpoints bind to the actual editable FK/PK rows.
ROUTES = {
    ('PRReviewComments', 'pr_id'): [(460, 394), (460, 220), (1010, 220), (1010, 658)],
    ('PRReviewComments', 'repo_id'): [(430, 430), (430, 180), (1510, 180)],
    ('PRReviewComments', 'author_id'): [(490, 466), (490, 250), (1450, 250)],
    ('PRReviewComments', 'pull_request_review_id'): [(480, 502), (480, 398)],
    ('PRReviewComments', 'in_reply_to_id'): [(20, 538), (20, 358)],
    ('IssueComments', 'issue_id'): [(480, 864), (480, 858)],
    ('IssueComments', 'repo_id'): [(440, 900), (440, 1190), (1490, 1190)],
    ('IssueComments', 'author_id'): [(460, 936), (460, 1040), (1470, 1040)],
    ('Pushes', 'user_id'): [(450, 1194), (450, 1300), (1430, 1300)],
    ('Pushes', 'repo_id'): [(480, 1230), (480, 1270), (1530, 1270)],
    ('PRReviews', 'pr_id'): [(1020, 434), (1020, 658)],
    ('PRReviews', 'repo_id'): [(960, 470), (960, 1100), (1450, 1100)],
    ('PRReviews', 'reviewer_id'): [(940, 506), (940, 280), (1430, 280)],
    ('Issues', 'repo_id'): [(940, 894), (940, 1140), (1470, 1140)],
    ('Issues', 'author_id'): [(960, 930), (960, 820), (1510, 820)],
    ('Issues', 'pull_request_id'): [(990, 966), (990, 658)],
    ('Issues', 'assignee_id'): [(1020, 1002), (1020, 850), (1490, 850)],
    ('PullRequests', 'repo_id'): [(1540, 694)],
    ('PullRequests', 'author_id'): [(1450, 730)],
    ('PullRequests', 'merged_by_id'): [(1470, 766)],
}


def key_id(table, column):
    return f'key__{table}__{column}'


def geometry(cell, **attributes):
    return ET.SubElement(cell, 'mxGeometry', **{k: str(v) for k, v in attributes.items()}, **{'as': 'geometry'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/question1')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    build = json.loads((ROOT / 'data/build_summary.json').read_text())
    start = datetime.fromisoformat(build['period_utc_inclusive'])
    end = datetime.fromisoformat(build['period_utc_exclusive'])
    hours = int((end - start).total_seconds() // 3600)
    assert hours == build['downloaded_hours']

    doc = ET.Element('mxfile', host='app.diagrams.net', compressed='false')
    page = ET.SubElement(doc, 'diagram', id='assignment2-schema', name='Schema Graph')
    model = ET.SubElement(page, 'mxGraphModel', dx=str(WIDTH), dy=str(HEIGHT),
                          grid='1', gridSize='10', guides='1', tooltips='1',
                          connect='1', arrows='1', fold='1', page='1', pageScale='1',
                          pageWidth=str(WIDTH), pageHeight=str(HEIGHT), math='0', shadow='0')
    root = ET.SubElement(model, 'root')
    ET.SubElement(root, 'mxCell', id='0')
    ET.SubElement(root, 'mxCell', id='1', parent='0')
    annotations = [
        ('title', 'Assignment 2 | Schema Graph', 60, 35, 1840, 45, 30),
        ('legend', '8 tables  |  8 primary keys (PK)  |  20 foreign keys (FK)  |  Arrows: FK -> referenced PK',
         60, 95, 1840, 32, 18),
        ('scope', 'Key columns shown. Each row and connector is editable; drag a table header to reposition its fields.',
         60, 132, 1840, 30, 16),
        ('period', f'Dataset (UTC): [{start:%Y-%m-%d %H:%M}, {end:%Y-%m-%d %H:%M})  |  {hours:,} hourly files',
         60, 1350, 1840, 30, 18),
        ('note', 'Anonymized code columns (e.g. github_user_id, org_id, head_repo_id) are not foreign keys.',
         60, 1390, 1840, 26, 15),
    ]
    for identifier, text, x, y, w, h, size in annotations:
        cell = ET.SubElement(root, 'mxCell', id=identifier, value=text, vertex='1', parent='1',
                             style=f'text;html=0;align=left;verticalAlign=middle;whiteSpace=wrap;'
                                   f'fontFamily=Helvetica;fontSize={size};fontColor=#263447;')
        geometry(cell, x=x, y=y, width=w, height=h)

    key_positions = {}
    for table, spec in TABLES.items():
        x, y = POSITIONS[table]
        fields = [(spec['pk'], 'PK')] + [(column, 'FK') for column in spec['fks']]
        row_height = 80 if len(fields) == 1 else ROW
        container = ET.SubElement(root, 'mxCell', id=f'table__{table}', value=table,
                                  vertex='1', parent='1',
                                  style='swimlane;horizontal=1;startSize=40;collapsible=0;'
                                        'container=1;recursiveResize=0;html=0;rounded=0;'
                                        f'fillColor={COLORS[table]};swimlaneFillColor=#ffffff;'
                                        'strokeColor=#718096;fontColor=#ffffff;fontSize=19;'
                                        'fontStyle=1;fontFamily=Helvetica;align=center;')
        geometry(container, x=x, y=y, width=TABLE_WIDTH, height=HEADER + len(fields) * row_height)
        for index, (column, kind) in enumerate(fields):
            offset = HEADER + index * row_height
            key_positions[(table, column)] = (x, y + offset, TABLE_WIDTH, row_height)
            cell = ET.SubElement(root, 'mxCell', id=key_id(table, column), value=f'[{kind}]  {column}',
                                 vertex='1', parent=f'table__{table}',
                                 style='rounded=0;html=0;whiteSpace=wrap;align=left;verticalAlign=middle;'
                                       'spacingLeft=12;fontSize=16;fontFamily=Helvetica;'
                                       f'fillColor={"#e8f0fb" if kind == "PK" else "#fff5df"};'
                                       'strokeColor=#cbd5e1;strokeWidth=0.6;fontColor=#162234;')
            geometry(cell, x=0, y=offset, width=TABLE_WIDTH, height=row_height)

    records = [{'source_table': table, 'foreign_key': column, 'target_table': target,
                'primary_key': TABLES[target]['pk']}
               for table, spec in TABLES.items() for column, target in spec['fks'].items()]
    incoming = Counter(r['target_table'] for r in records)
    allocated = defaultdict(int)
    preview_edges = []
    for record in records:
        table, column, target, pk = (record[k] for k in ['source_table', 'foreign_key', 'target_table', 'primary_key'])
        allocated[target] += 1
        fraction = allocated[target] / (incoming[target] + 1)
        sx, sy, sw, sh = key_positions[(table, column)]
        tx, ty, tw, th = key_positions[(target, pk)]
        self_loop = table == target
        source_point = (sx if self_loop else sx + sw, sy + sh / 2)
        target_point = (tx, ty + th * fraction)
        waypoints = list(ROUTES[(table, column)])
        if waypoints[-1][1] != target_point[1]:
            waypoints.append((waypoints[-1][0], target_point[1]))
        edge = ET.SubElement(root, 'mxCell', id=f'fk__{table}__{column}', edge='1', parent='1',
                             source=key_id(table, column), target=key_id(target, pk),
                             style='edgeStyle=segmentEdgeStyle;rounded=1;html=0;'
                                   'endArrow=block;endFill=1;endSize=8;startArrow=none;'
                                   f'strokeColor={COLORS[target]};strokeWidth=1.6;'
                                   f'exitX={0 if self_loop else 1};exitY=0.5;exitPerimeter=0;'
                                   f'entryX=0;entryY={fraction:.6f};entryPerimeter=0;'
                                   'jumpStyle=arc;jumpSize=8;')
        points = ET.SubElement(geometry(edge, relative=1), 'Array', **{'as': 'points'})
        for px, py in waypoints:
            ET.SubElement(points, 'mxPoint', x=str(px), y=str(py))
        preview_edges.append(([source_point, *waypoints, target_point], COLORS[target]))

    ET.indent(doc, space='  ')
    native = args.out / 'schema.drawio'
    ET.ElementTree(doc).write(native, encoding='utf-8', xml_declaration=True)
    cells = ET.parse(native).findall('.//mxCell')
    groups = [c for c in cells if c.get('id', '').startswith('table__')]
    edges = [c for c in cells if c.get('edge') == '1']
    expected = {(key_id(r['source_table'], r['foreign_key']),
                 key_id(r['target_table'], r['primary_key'])) for r in records}
    assert len(groups) == 8 and len(edges) == 20
    assert {(e.get('source'), e.get('target')) for e in edges} == expected
    assert not any('image=' in c.get('style', '') for c in cells)

    # Previews use the same initial geometry; the editable master remains the .drawio file.
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.set_axis_off()
    for points, color in preview_edges:
        ax.add_patch(FancyArrowPatch(path=PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1)),
                                     arrowstyle='-|>', mutation_scale=11, color=color, linewidth=1.2, zorder=1))
    for table, spec in TABLES.items():
        x, y = POSITIONS[table]
        ax.add_patch(Rectangle((x, y), TABLE_WIDTH, HEADER, facecolor=COLORS[table], edgecolor='#718096', zorder=2))
        ax.text(x + TABLE_WIDTH / 2, y + HEADER / 2, table, va='center', ha='center',
                fontsize=14, color='white', weight='bold', zorder=3)
        for column, kind in [(spec['pk'], 'PK')] + [(c, 'FK') for c in spec['fks']]:
            rx, ry, rw, rh = key_positions[(table, column)]
            ax.add_patch(Rectangle((rx, ry), rw, rh, facecolor='#e8f0fb' if kind == 'PK' else '#fff5df',
                                   edgecolor='#cbd5e1', linewidth=.6, zorder=2))
            ax.text(rx + 12, ry + rh / 2, f'[{kind}]  {column}', va='center', ha='left', fontsize=11.4, zorder=3)
    for _, text, x, y, w, h, size in annotations:
        ax.text(x, y + h / 2, text, va='center', ha='left', fontsize=size * .72, color='#263447', zorder=3)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(args.out / f'schema.preview.{extension}', dpi=150, facecolor='white')
    plt.close(fig)
    with (args.out / 'foreign_keys.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    last_date = (end - timedelta(microseconds=1)).date()
    start_tw, end_tw = (t.astimezone(ZoneInfo('Asia/Taipei')) for t in (start, end))
    answer = f'''# Question 1: Schema Graph and Dataset Time Period

## Schema graph

The schema graph contains all **8 tables**, with **8 primary keys (PK)** and
**20 foreign keys (FK)**. Every arrow runs from a foreign-key column to the
referenced primary-key column (**FK → PK**). All key columns are marked in their table nodes.

![Schema graph: 8 tables and all 20 FK-to-PK edges](schema.preview.png)

Editable source: [schema.drawio](schema.drawio). This file contains native draw.io
table containers, editable key rows, and individually connected FK edges.
The PNG/SVG/PDF files are previews of the initial layout; after editing the master,
export a new image from draw.io for submission.

The self-referencing edge is
`PRReviewComments.in_reply_to_id → PRReviewComments.comment_id`.
Parallel relationships remain separate edges, including both PR author/merger and
Issue author/assignee references to `Users.user_id`. Anonymized code columns such as
`github_user_id`, `org_id`, and `head_repo_id` are not foreign keys.

## Dataset time period

The dataset selected for model development covers **{start:%Y-%m-%d} through
{last_date:%Y-%m-%d}, inclusive, in UTC**: **{hours:,} hourly GH Archive files**
spanning **{(end - start).days} complete UTC days**. The provided `scripts/preprocess.py`
was used to generate the eight anonymized tables.

| Time zone | Start (inclusive) | End (exclusive) |
|---|---|---|
| UTC | {start:%Y-%m-%d %H:%M:%S} | {end:%Y-%m-%d %H:%M:%S} |
| Taiwan (UTC+8) | {start_tw:%Y-%m-%d %H:%M:%S} | {end_tw:%Y-%m-%d %H:%M:%S} |

For a prediction at time `t`, only rows with **row time < t** are eligible.
The row time is `first_seen_at` for Users, Repositories, PullRequests, Issues,
and PRReviews, and `created_at` for IssueComments, PRReviewComments, and Pushes.

Sources: [official schema](../../SCHEMA.md), [assignment instructions](../../README.md),
and [dataset build record](../../data/build_summary.json).
The full [foreign-key list](foreign_keys.csv) contains one row per edge.

## 在 draw.io 微調

1. 開啟 [app.diagrams.net](https://app.diagrams.net/)，選擇 Device，再開啟 `schema.drawio`；
   也可把檔案拖進空白畫布，或使用 draw.io Desktop。
2. 拖動表格標題可移動整張表；雙擊標題或欄位可修改文字。
3. 點選連線後拖動轉折控制點可微調路徑；線的起點／終點已連到對應的 FK／PK 欄位。
4. 保留 `.drawio` 作為可編輯原稿，再從 File → Export as 匯出 PNG、SVG 或 PDF。

若需重建初始版本，在專案根目錄執行 `python scripts/draw_schema_drawio.py`。
重建會覆寫 `schema.drawio`，因此微調後請另存副本，避免覆蓋手動調整。
'''
    (args.out / 'question1.md').write_text(answer)
    print(f'PASS: {native}; 8 native tables, 8 PK rows, 20 connected FK edges, no embedded bitmap.')


if __name__ == '__main__':
    main()

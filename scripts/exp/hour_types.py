"""Count GH Archive event types per hourly file (raw originals, no filtering). usage: hour_types.py OUT.csv DIR_OR_FILES..."""
import sys, gzip, re, glob, csv, collections
from multiprocessing import Pool
T = re.compile(rb'"type":"([A-Za-z]+)"')
def count(path):
    c = collections.Counter(); n = 0
    with gzip.open(path, 'rb') as f:
        for line in f:
            m = T.search(line)
            if m: c[m.group(1).decode()] += 1
            n += 1
    name = path.split('/')[-1][:-8]
    return name, n, c
if __name__ == '__main__':
    out = sys.argv[1]; files = []
    for a in sys.argv[2:]: files += sorted(glob.glob(a + '/*.json.gz')) if not a.endswith('.gz') else [a]
    with Pool(4) as p, open(out, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['hour', 'events', 'type', 'count'])
        for name, n, c in p.imap(count, files):
            for t, k in c.items(): w.writerow([name, n, t, k])

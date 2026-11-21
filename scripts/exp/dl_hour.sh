#!/bin/bash
f=$1; out=/home/asiadragon/geolife/bigdata_asg/BigData2026-Assignment2/data/y2025dec/raw
[ -s $out/$f.json.gz ] && gzip -t $out/$f.json.gz 2>/dev/null && exit 0
for i in 1 2 3; do curl -sfL -m 300 -A "gharchive-download/1.0" -o $out/$f.json.gz.part https://data.gharchive.org/$f.json.gz && gzip -t $out/$f.json.gz.part && mv $out/$f.json.gz.part $out/$f.json.gz && exit 0; done
echo FAIL $f

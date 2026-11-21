"""Download full GH Archive hours, verify gzip, and keep an auditable manifest.

Dates are inclusive UTC dates. No filtering or anonymization happens here:
the downloaded originals must be passed through the official preprocess.py.
Existing valid files are reused; partial downloads never get a .json.gz name.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import shutil
import time
import zlib
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def valid_gzip(path):
    try:
        if path.stat().st_size == 0:
            return False
        with gzip.open(path, "rb") as stream:
            while stream.read(1024 * 1024):
                pass
        return True
    except (OSError, EOFError, zlib.error):
        return False


def download(hour, raw, retries):
    filename = f"{hour:%Y-%m-%d}-{hour.hour}.json.gz"
    url = f"https://data.gharchive.org/{filename}"
    target = raw / filename
    record = {"hour_utc": hour.isoformat(), "file": filename, "url": url}
    if target.exists() and valid_gzip(target):
        return dict(record, status="existing", bytes=target.stat().st_size)
    partial = target.with_suffix(target.suffix + ".part")
    error = ""
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "BigData2026-Assignment2-setup/1.0"})
            with urlopen(request, timeout=90) as response, partial.open("wb") as output:
                expected = response.headers.get("Content-Length")
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if expected and partial.stat().st_size != int(expected):
                raise OSError("downloaded size differs from Content-Length")
            if not valid_gzip(partial):
                raise OSError("gzip integrity check failed")
            partial.replace(target)
            return dict(record, status="downloaded", bytes=target.stat().st_size)
        except HTTPError as exc:
            error = str(exc)
            if exc.code == 404:
                partial.unlink(missing_ok=True)
                return dict(record, status="missing", error=error)
        except (OSError, URLError, TimeoutError) as exc:
            error = str(exc)
        if attempt < retries:
            time.sleep(min(2 ** attempt, 8))
    partial.unlink(missing_ok=True)
    return dict(record, status="failed", error=error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 10, 1))
    parser.add_argument("--end", type=date.fromisoformat,
                        default=datetime.now(timezone.utc).date() - timedelta(days=1),
                        help="inclusive UTC date; default: latest completed UTC day")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/gharchive"))
    parser.add_argument("--manifest", type=Path, default=Path("data/download_manifest.json"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.start > args.end or args.start < date(2025, 12, 1):
        parser.error("require 2025-12-01 <= start <= end")
    if args.workers < 1 or args.retries < 0:
        parser.error("workers must be positive and retries non-negative")
    begin = datetime.combine(args.start, datetime.min.time(), tzinfo=timezone.utc)
    stop = datetime.combine(args.end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    if stop > datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0):
        parser.error("end must be a completed UTC date")
    hours = [begin + timedelta(hours=i) for i in range(int((stop - begin).total_seconds() // 3600))]
    print(f"UTC [{begin.isoformat()}, {stop.isoformat()}): {len(hours)} hours -> {args.raw}", flush=True)
    if args.dry_run:
        return
    args.raw.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    records = []
    counts = {}
    def save_manifest():
        summary = {
            "start_utc_inclusive": begin.isoformat(), "end_utc_exclusive": stop.isoformat(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "expected_hours": len(hours), "completed_hours": len(records), "counts": counts,
            "files": sorted(records, key=lambda r: r["hour_utc"]),
        }
        temporary = args.manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(args.manifest)
    save_manifest()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, hour, args.raw, args.retries) for hour in hours]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            status = record["status"]
            counts[status] = counts.get(status, 0) + 1
            if status in ("failed", "missing"):
                print(f"{status}: {record['file']}: {record['error']}", flush=True)
            if len(records) % 24 == 0 or len(records) == len(hours):
                save_manifest()
                print(f"{len(records)}/{len(hours)} hours: {counts}", flush=True)
    if counts.get("failed") or counts.get("missing"):
        raise SystemExit("Incomplete coverage; inspect the manifest and rerun to retry missing hours.")


if __name__ == "__main__":
    main()

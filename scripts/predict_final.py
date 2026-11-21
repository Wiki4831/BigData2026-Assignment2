"""Final submission pipeline: train the best recipe and write output_{studentID}.csv.

Recipe: RelBench GraphSAGE encoder (3 hops) -> pretrain 3 epochs on next-day BCE -> freeze -> 7-step GRU
with daily BCE and a per-user recent-statistics module -> weekly probability = max of the 7 daily
probabilities -> per-activity thresholds that maximise the validation score -> average over seeds.

Data gap: GH Archive files end before the forecast cutoff (the "gap"). Every training, validation and
forecast sample is built with the same gap: rows are visible only before (cutoff - gap), but the
relative-time encoder and the history statistics measure ages from the cutoff. The validation week is
the latest 7-day Taiwan-midnight window that ends inside the data.

Rows at or after the cutoff are never used, even if the downloaded files contain them.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

if "--deterministic" in sys.argv:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch_geometric import seed_everything
from torch_geometric.loader import NeighborLoader

import rdl_model as R
from evaluate import EVENT_COLS, OUTPUT_COLUMNS, USER_COL, read_predictions
from gnn_model import NextDayActivityModel
from model_data import ActivityData, Coverage, DAY, WEEK, hour_boundary, taipei_date, utc_timestamp
from common import ROOT, choose_thresholds, evaluate_week, training_cutoffs, training_labels, write_atomically
from user_history import HistoryExtras, HistoryInputs


class GapHistoryInputs(HistoryInputs):
    """HistoryInputs.stats with a visibility bound that is earlier than the cutoff used for ages."""

    def stats(self, cutoff, gap=pd.Timedelta(0)):
        if not gap:
            return super().stats(cutoff)
        c = np.int64(cutoff.to_datetime64().astype("datetime64[s]").astype(np.int64))
        bound = c - np.int64(gap.total_seconds())
        vis = (self.t < bound) & (self.r < bound)
        # Same statistics as HistoryInputs.stats, with ages measured from c.
        from user_history import N_ACT, WINDOWS
        u, a, age = self.u[vis], self.act[vis], (c - self.t[vis]) / 86400.0
        cols = []
        for k in range(N_ACT):
            m = a == k
            uk, ak = u[m], age[m]
            for w in WINDOWS:
                cols.append(np.log1p(np.bincount(uk[ak <= w], minlength=self.n)))
            last = np.full(self.n, np.nan)
            np.fmin.at(last, uk, ak)
            cols.append(np.log1p(np.nan_to_num(last, nan=60.0)))
            day = np.floor(ak).astype(int)
            keep = day < 28
            pairs = np.unique(uk[keep].astype(np.int64) * 28 + day[keep])
            cols.append(np.bincount(pairs // 28, minlength=self.n) / 28.0)
        day = np.floor(age).astype(int)
        keep = day < 28
        pairs = np.unique(u[keep].astype(np.int64) * 28 + day[keep])
        cols.append(np.bincount(pairs // 28, minlength=self.n) / 28.0)
        cols.append(np.log1p(np.bincount(u, minlength=self.n)))
        return np.column_stack(cols).astype(np.float32)


def make_view(data, cutoff, user_ids, gap, *, labeled, horizon=WEEK, daily=False):
    """Users' graph membership as seen at (cutoff - gap), plus labels of the window starting at cutoff."""
    view = SimpleNamespace(cutoff=cutoff, user_nodes=data.user_nodes(cutoff - gap, user_ids), target_counts=None,
                           label_mask=None, next_day_counts=None, next_day_mask=None)
    if labeled:
        counts, mask = data._targets(cutoff, pd.Index(user_ids, dtype="int64"), by_day=daily, horizon=horizon)
        if daily:
            view.next_day_counts, view.next_day_mask = counts, mask
            counts = counts.sum(axis=1)
            mask = (counts > 0) | view.next_day_mask.all(axis=1)
        view.target_counts, view.label_mask = counts, mask
    return view


class Attach:
    """Attach labels and history statistics to a batch and move the seed time forward by the gap."""

    def __init__(self, target, stats, gap_seconds):
        self.target, self.stats, self.gap_seconds = target, stats, gap_seconds

    def __call__(self, batch):
        store = batch[R.ENTITY]
        index = store.input_id
        if self.target is not None:
            store.y = self.target[index]
        if self.stats is not None:
            store.hist_stats = self.stats[index]
        store.seed_time = store.seed_time + self.gap_seconds
        return batch


def unix_seconds(timestamp):
    return int(timestamp.to_datetime64().astype("datetime64[s]").astype(np.int64))


def make_loader(graph, data, user_ids, views, cold_id, args, gap, shuffle, hist=None):
    codes = data.user_codes
    original_ids = pd.Series(codes.index, index=codes.to_numpy()).reindex(user_ids)
    nodes, times, targets, masks, stats = [], [], [], [], []
    for view in views:
        nodes.append(original_ids.where(view.user_nodes.numpy() >= 0, cold_id).to_numpy(dtype=np.int64))
        # Sampling is inclusive and the cutoff is exclusive; sample as of the last visible second.
        times.append(np.full(len(user_ids), unix_seconds(view.cutoff - gap) - 1, dtype=np.int64))
        if view.target_counts is not None:
            counts, mask = training_labels(view)
            targets.append((counts > 0).astype(np.float32))
            masks.append(mask)
        if hist is not None:
            stats.append(hist.stats(view.cutoff, gap))
    target = torch.from_numpy(np.concatenate(targets)) if targets else None
    transform = Attach(target, torch.from_numpy(np.concatenate(stats)) if stats else None,
                       int(gap.total_seconds()))
    loader = NeighborLoader(graph, num_neighbors=R.neighbors_by_hop(args.hops), time_attr="time",
                            input_nodes=(R.ENTITY, torch.from_numpy(np.concatenate(nodes))),
                            input_time=torch.from_numpy(np.concatenate(times)), transform=transform,
                            temporal_strategy="uniform", batch_size=args.batch_size,
                            shuffle=shuffle, num_workers=0)
    mask = torch.from_numpy(np.concatenate(masks)).bool() if masks else None
    return loader, mask


def run_seed(seed, args, graph, column_stats, data, user_ids, cold_id, hist, gap, val_time, views, device, upstream_model):
    seed_everything(seed)
    graph_model = upstream_model(data=graph, col_stats_dict=column_stats, num_layers=args.hops,
                                 channels=R.CHANNELS, out_channels=len(EVENT_COLS), aggr="sum",
                                 norm="batch_norm", gnn="sage")
    history = R.RelBenchActivityHistory(graph_model, cold_id).to(device)

    # Stage 1: pretrain the encoder on next-day labels with cutoffs whose day ends before validation.
    day_cutoffs = training_cutoffs(data.coverage, val_time, args.min_history_hours, stride=DAY, target_horizon=DAY)
    pre_views = [make_view(data, c, user_ids, gap, labeled=True, horizon=DAY) for c in day_cutoffs]
    loader, mask = make_loader(graph, data, user_ids, pre_views, cold_id, args, gap, True)
    del pre_views
    R.initialize_loader_prior(history, loader, mask)
    optimizer = torch.optim.AdamW(history.parameters(), lr=R.DAILY_LEARNING_RATE)
    pre_log = []
    for epoch in range(1, args.pretrain_epochs + 1):
        started = perf_counter()
        loss = R.train_epoch(history, optimizer, loader, mask, device)
        pre_log.append({"epoch": epoch, "loss": loss, "seconds": perf_counter() - started})
        print(json.dumps({"seed": seed, "stage": "pretrain", **pre_log[-1]}), flush=True)
    del loader, mask
    history.freeze_encoder()

    # Stage 2: seven-step GRU with the daily loss; the encoder stays frozen.
    model = NextDayActivityModel(history, R.DAILY_HIDDEN_SIZE, len(EVENT_COLS)).to(device)
    if args.history != "none":
        model.extras = HistoryExtras(R.DAILY_HIDDEN_SIZE, cold_id).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=R.DAILY_LEARNING_RATE)
    train_cutoffs = training_cutoffs(data.coverage, val_time, args.min_history_hours, stride=DAY)
    train_views = [make_view(data, c, user_ids, gap, labeled=True, daily=True) for c in train_cutoffs]
    train_loader, train_mask = make_loader(graph, data, user_ids, train_views, cold_id, args, gap, True, hist)
    del train_views
    val_loader, _ = make_loader(graph, data, user_ids, [views["validation"]], cold_id, args, gap, False, hist)
    test_loader, _ = make_loader(graph, data, user_ids, [views["forecast"]], cold_id, args, gap, False, hist)

    best_score, best_epoch, best = float("-inf"), 0, None
    epochs = []
    for epoch in range(1, args.epochs + 1):
        started = perf_counter()
        loss = R.train_epoch(model, optimizer, train_loader, train_mask, device)
        val_probabilities = R.predict(model, val_loader, device)
        _, score = choose_thresholds(val_probabilities, views["validation"])
        epochs.append({"epoch": epoch, "loss": loss, "validation_score": score, "seconds": perf_counter() - started})
        print(json.dumps({"seed": seed, "stage": "gru", **epochs[-1]}), flush=True)
        if score > best_score + 1e-12:
            best_score, best_epoch = score, epoch
            test_probabilities = R.predict(model, test_loader, device)
            best = (val_probabilities, test_probabilities)
        elif epoch - best_epoch >= args.patience:
            break
    return {"seed": seed, "val": best[0], "test": best[1], "best_epoch": best_epoch,
            "best_validation_score": best_score, "pretrain": pre_log, "epochs": epochs,
            "train_cutoffs": [taipei_date(c) for c in train_cutoffs],
            "pretrain_cutoffs": [taipei_date(c) for c in day_cutoffs]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student-id", required=True, help="Used in the default output name output_{id}.csv.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workdir", type=Path, default=ROOT / "data/model/final")
    parser.add_argument("--db", type=Path, default=ROOT / "data/processed/db")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/download_manifest.json")
    parser.add_argument("--cohort", type=Path, default=ROOT / "scripts/cohort.txt")
    parser.add_argument("--test-input", type=Path, default=ROOT / "submission/test_input.csv")
    parser.add_argument("--upstream", type=Path, default=R.RUN_DIR / "upstream")
    parser.add_argument("--forecast-cutoff", default="2026-11-22",
                        help="Start of the forecast week. A date without a zone is Taiwan time.")
    parser.add_argument("--available-until",
                        help="Backtest only: ignore data from this UTC time on, so the gap is simulated.")
    parser.add_argument("--gap-hours", type=int, help="Default: forecast cutoff minus the last usable data hour.")
    parser.add_argument("--validation-cutoff", default="auto",
                        help="Taiwan date of the validation week start. Default: the latest 7-day Taiwan-midnight "
                             "window that ends inside the usable data.")
    parser.add_argument("--score-forecast", action="store_true",
                        help="Backtest only: the forecast week has labels, report its score.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 57, 2026])
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--min-history-hours", type=int, default=24)
    parser.add_argument("--history", choices=["none", "stats"], default="stats")
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--threads", type=int)
    parser.add_argument("--deterministic", action="store_true",
                        help="Ask PyTorch for deterministic kernels (slower; unsupported ops only warn).")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.hops, args.pretrain_epochs, len(args.seeds)) < 1:
        parser.error("Epochs, pretrain epochs, batch size, patience, hops and seeds must be positive.")

    sys.path.insert(0, str(args.upstream.resolve() / "examples"))
    from model import Model
    from text_embedder import GloveTextEmbedding

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(args.threads or (1 if device.type == "cuda" else 4))
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    forecast = hour_boundary(args.forecast_cutoff)
    coverage = Coverage.read(args.manifest)
    # Last instant whose data may be used: the end of the files, never later than the cutoff.
    available = min(coverage.end, forecast)
    if args.available_until:
        available = min(available, hour_boundary(args.available_until))
    gap = pd.Timedelta(hours=args.gap_hours) if args.gap_hours is not None else forecast - available
    if gap < pd.Timedelta(0):
        parser.error("The gap cannot be negative.")
    if args.validation_cutoff == "auto":
        local = (available + pd.Timedelta(hours=8) - WEEK).floor("D")
        val_time = local - pd.Timedelta(hours=8)
    else:
        val_time = hour_boundary(args.validation_cutoff)
    if val_time + WEEK > available:
        parser.error("The validation week must end inside the usable data.")
    if args.score_forecast and forecast + WEEK > coverage.end:
        parser.error("--score-forecast needs labels for the whole forecast week.")
    print(f"forecast {taipei_date(forecast)} | data usable before {available} UTC | gap {gap} | "
          f"validation week {taipei_date(val_time)} | seeds {args.seeds} | {device}", flush=True)

    user_ids = np.loadtxt(args.cohort, dtype=np.int64, ndmin=1)
    test_ids = pd.read_csv(args.test_input, dtype="int64")[USER_COL].to_numpy()
    if len(np.unique(user_ids)) != len(user_ids) or len(np.unique(test_ids)) != len(test_ids):
        raise ValueError("Cohort and test IDs must be unique.")
    data = ActivityData(args.db, coverage)
    hist = GapHistoryInputs(data.events, user_ids) if args.history != "none" else None
    database, cold_id = R.read_database(args.db, available, coverage.start)
    graph, column_stats, _ = R.make_graph(database, GloveTextEmbedding, fit_before=val_time)
    print(f"{sum(graph.num_nodes_dict.values()):,} nodes | {graph.num_edges:,} directed edges", flush=True)

    views = {"validation": make_view(data, val_time, user_ids, gap, labeled=True),
             "forecast": make_view(data, forecast, user_ids, gap, labeled=args.score_forecast)}
    runs = [run_seed(seed, args, graph, column_stats, data, user_ids, cold_id, hist, gap, val_time, views,
                     device, Model) for seed in args.seeds]

    val_probabilities = np.mean([r["val"] for r in runs], axis=0)
    test_probabilities = np.mean([r["test"] for r in runs], axis=0)
    thresholds, val_score = choose_thresholds(val_probabilities, views["validation"])
    predictions = test_probabilities >= thresholds

    # Map the test list to cohort rows. A test user outside the cohort has no graph node and gets the
    # shared cold-start prediction, the same as any cold cohort user.
    position = pd.Series(np.arange(len(user_ids)), index=user_ids).reindex(test_ids)
    warm = views["forecast"].user_nodes.numpy() >= 0
    outside = position.isna().to_numpy()
    if outside.any():
        cold_rows = np.flatnonzero(~warm)
        if not len(cold_rows):
            raise ValueError("Some test users are outside the cohort and no cold user exists to copy.")
        position[outside] = cold_rows[0]
    rows = position.to_numpy(dtype=np.int64)
    output = pd.DataFrame(predictions[rows].astype(int), columns=EVENT_COLS)
    output.insert(0, USER_COL, test_ids)
    output_path = args.output or ROOT / f"output_{args.student_id}.csv"
    write_atomically(output_path, lambda path: output.to_csv(path, index=False))
    read_predictions(output_path, expected_users=test_ids)  # same format check as scripts/evaluate.py

    args.workdir.mkdir(parents=True, exist_ok=True)
    tag = output_path.stem
    np.savez_compressed(args.workdir / f"{tag}.probs.npz", val=val_probabilities, test=test_probabilities,
                        seeds=np.array(args.seeds), user_ids=user_ids, thresholds=thresholds,
                        warm_forecast=warm, val_target=views["validation"].target_counts > 0)
    sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
    revision = subprocess.check_output(["git", "-C", args.upstream.resolve(), "rev-parse", "HEAD"], text=True).strip()
    report = {
        "output": str(output_path), "output_sha256": sha, "rows": len(output),
        "forecast_week_taipei": taipei_date(forecast), "data_usable_before_utc": str(available),
        "gap_hours": gap / pd.Timedelta(hours=1), "validation_week_taipei": taipei_date(val_time),
        "seeds": args.seeds, "thresholds": dict(zip(EVENT_COLS, thresholds.tolist())),
        "validation_ensemble_score": val_score,
        "validation": evaluate_week(val_probabilities >= thresholds, views["validation"]),
        "per_seed": [{k: v for k, v in r.items() if k not in ("val", "test")} for r in runs],
        "forecast_warm_users": int(warm.sum()), "test_users_outside_cohort": int(outside.sum()),
        "predicted_positive_rate_test": dict(zip(EVENT_COLS, output[EVENT_COLS].mean().round(4).tolist())),
        "true_positive_rate_validation": dict(zip(
            EVENT_COLS, (views["validation"].target_counts > 0).mean(axis=0).round(4).tolist())),
        "relbench_commit": revision, "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "versions": {n: version(n) for n in ("relbench", "pytorch-frame", "torch", "torch-geometric", "pyg-lib")},
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    if args.score_forecast:
        report["forecast_score"] = evaluate_week(predictions, views["forecast"])
        report["forecast_score_warm"] = evaluate_week(predictions, views["forecast"], warm)
    write_atomically(args.workdir / f"{tag}.json", lambda path: path.write_text(json.dumps(report, indent=2) + "\n"))
    print(json.dumps({k: report[k] for k in ("validation_ensemble_score", "thresholds",
                                              "predicted_positive_rate_test", "output_sha256")}, indent=2))
    if args.score_forecast:
        print(json.dumps(report["forecast_score"], indent=2))


if __name__ == "__main__":
    main()

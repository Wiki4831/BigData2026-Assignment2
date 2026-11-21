"""Compare equal-budget calendar and event-triggered RelBench pretraining."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from relbench.modeling.graph import AttachTargetTransform
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from torch_geometric import seed_everything
from torch_geometric.loader import NeighborLoader

from evaluate import EVENT_COLS
from gnn_model import NextDayActivityModel
from model_data import ActivityData, Coverage, DAY, WEEK, hour_boundary, taipei_date
from relbench_asg2 import (CHANNELS, DAILY_HIDDEN_SIZE, DAILY_LEARNING_RATE, ENTITY, RUN_DIR,
                           RelBenchActivityHistory, initialize_loader_prior, make_graph, make_loader,
                           neighbors_by_hop, predict, read_database, train_epoch)
from relbench_contrastive import train_contrastive_epoch
from relbench_masked import first_visible_relationship
from train_model import ROOT, choose_thresholds, evaluate_week, known_labels, training_cutoffs, write_atomically


VARIANTS = ((24, 0), (12, 0), (12, 20), (12, 40), (6, 20))
HOPS = 3
SECS_PER_HOUR = 3600


class Samples(NamedTuple):
    users: np.ndarray
    time: np.ndarray
    targets: np.ndarray
    known: np.ndarray


def calendar_samples(data, users, start, end, hours, count, seed):
    duration = pd.Timedelta(hours=hours)
    cutoffs = pd.date_range(start, end - duration, freq=duration)
    population = len(users) * len(cutoffs)
    selected = np.sort(np.random.default_rng(seed + hours).choice(population, size=count, replace=False))
    cutoff_ids, user_ids = divmod(selected, len(users))
    targets = np.zeros((count, len(EVENT_COLS)), dtype=bool)
    known = np.zeros_like(targets)
    left = np.searchsorted(cutoff_ids, np.arange(len(cutoffs) + 1))
    for i, cutoff in enumerate(cutoffs):
        block = slice(left[i], left[i + 1])
        positive, observed = data._targets(cutoff, users[user_ids[block]], horizon=duration)
        targets[block], known[block] = positive > 0, observed
    times = cutoffs.to_numpy(dtype="datetime64[s]").astype(np.int64)[cutoff_ids]
    return Samples(user_ids, times, targets, known)


def event_samples(data, users, start, end, horizons):
    events = data.events.loc[data.events["github_user_id"].isin(users)].copy()
    events["visible"] = events[["event_time", "row_time"]].max(axis=1)
    events["position"] = users.get_indexer(events["github_user_id"])
    start_time = start.value // 1_000_000_000
    end_time = end.value // 1_000_000_000
    spacing = int(DAY.total_seconds())
    windows = {hours: [] for hours in horizons}
    subject, cutoffs = [], []
    for user, rows in events.groupby("position", sort=True):
        visible = np.unique(rows["visible"].to_numpy(dtype="datetime64[s]").astype(np.int64))
        ordered = rows.sort_values("event_time")
        times = ordered["event_time"].to_numpy(dtype="datetime64[s]").astype(np.int64)
        row_times = ordered["row_time"].to_numpy(dtype="datetime64[s]").astype(np.int64)
        labels = ordered["activity"].to_numpy()
        anchor = np.searchsorted(visible, start_time)
        while anchor < len(visible):
            # One second later includes the trigger in history but excludes it from targets.
            cutoff = int(visible[anchor]) + 1
            if cutoff + spacing > end_time:
                break
            subject.append(user)
            cutoffs.append(cutoff)
            left = np.searchsorted(times, cutoff)
            for hours, targets in windows.items():
                deadline = cutoff + hours * SECS_PER_HOUR
                right = np.searchsorted(times, deadline)
                target = np.zeros(len(EVENT_COLS), dtype=bool)
                target[labels[left:right][row_times[left:right] < deadline]] = True
                targets.append(target)
            anchor = np.searchsorted(visible, cutoff + spacing - 1)

    cutoffs = np.asarray(cutoffs, dtype=np.int64)
    subject = np.asarray(subject, dtype=np.int64)
    coverage = data.coverage.activity_hours.to_numpy(dtype=np.int64)
    cumulative = np.vstack([np.zeros((1, len(EVENT_COLS)), dtype=np.int64), coverage.cumsum(axis=0)])
    start_hour = data.coverage.start.value // (1_000_000_000 * SECS_PER_HOUR)
    first_hour = cutoffs // SECS_PER_HOUR - start_hour
    samples = {}
    for hours, values in windows.items():
        # Fractional windows can overlap both boundary archive hours.
        end_hour = (cutoffs + hours * SECS_PER_HOUR + SECS_PER_HOUR - 1) // SECS_PER_HOUR - start_hour
        covered = cumulative[end_hour] - cumulative[first_hour] == (end_hour - first_hour)[:, None]
        targets = np.asarray(values, dtype=bool)
        samples[hours] = Samples(subject, cutoffs, targets, covered | targets)
    return samples


def pretrain_loader(graph, samples, original_ids, first_relationship, cold_id, batch_size, *, shuffle=True):
    nodes = original_ids[samples.users].copy()
    in_graph = (nodes >= 0) & (nodes < cold_id)
    known = in_graph & (first_relationship[np.where(in_graph, nodes, cold_id)] < samples.time)
    nodes[~known] = cold_id
    loader = NeighborLoader(graph, num_neighbors=neighbors_by_hop(HOPS), input_nodes=(ENTITY, torch.from_numpy(nodes)),
                            input_time=torch.from_numpy(samples.time - 1), time_attr="time",
                            temporal_strategy="uniform", batch_size=batch_size, shuffle=shuffle, num_workers=0,
                            transform=AttachTargetTransform(ENTITY, torch.from_numpy(samples.targets).float()))
    return loader, torch.from_numpy(samples.known)


def encoder_validation(history, loader, samples, device, seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        probabilities = predict(history, loader, device).astype(np.float64).clip(1e-9, 1 - 1e-9)
    targets, known = samples.targets, samples.known
    active = known.all(axis=1) & targets.any(axis=1)
    loss = np.where(targets, -np.log(probabilities), -np.log1p(-probabilities))
    by_activity = {}
    for i, name in enumerate(EVENT_COLS):
        truth = targets[active, i]
        if truth.any() and not truth.all():
            by_activity[name] = float(average_precision_score(truth, probabilities[active, i]))
    return {"bce": float(loss[known].mean()), "active_samples": int(active.sum()),
            "active_macro_ap": float(np.mean(list(by_activity.values()))) if by_activity else None,
            "active_ap_by_activity": by_activity}


@torch.no_grad()
def cache_corrections(history, loader, device, seed):
    history.eval()
    corrections = torch.full((len(loader.input_data.node), DAILY_HIDDEN_SIZE), float("nan"), device=device)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for batch in loader:
            idx = batch[ENTITY].input_id.to(device)
            corrections[idx] = history.user_correction(batch.to(device))
    if not torch.isfinite(corrections).all():
        raise ValueError("Every forecast sample must receive a finite user correction.")
    return corrections


@torch.no_grad()
def forecast_probabilities(model, vectors, batch_size):
    model.eval()
    return torch.cat([model.forecast_from_correction(batch).sigmoid().amax(dim=1).cpu()
                      for batch in vectors.split(batch_size)]).numpy()


def fit_gru(model, train_vectors, train_y, train_known, val_vectors, validation, args, device):
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=DAILY_LEARNING_RATE)
    random = torch.Generator(device=device).manual_seed(args.seed)
    observed = train_known.flatten(start_dim=1).any(dim=1).nonzero(as_tuple=True)[0]
    best_score = -np.inf
    records = []
    for epoch in range(1, args.epochs + 1):
        started = perf_counter()
        model.train()
        total_loss = torch.zeros((), device=device)
        label_count = torch.zeros((), device=device)
        shuffled = observed[torch.randperm(len(observed), generator=random, device=device)]
        for indices in shuffled.split(args.gru_batch_size):
            targets, known = train_y[indices], train_known[indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model.forecast_from_correction(train_vectors[indices])
            losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss_sum = (losses * known).sum()
            labels = known.sum()
            (loss_sum / labels.clamp(min=1)).backward()
            optimizer.step()
            total_loss += loss_sum.detach()
            label_count += labels

        probabilities = forecast_probabilities(model, val_vectors, args.gru_batch_size)
        thresholds, score = choose_thresholds(probabilities, validation)
        record = {"epoch": epoch, "loss": (total_loss / label_count).item(),
                  "validation_score": score, "seconds": perf_counter() - started}
        records.append(record)
        print(json.dumps({"stage": "gru", **record}), flush=True)
        if score > best_score + 1e-12:
            best_score, best_epoch = score, epoch
            best_weights = copy.deepcopy(model.state_dict())
            best_thresholds, best_probabilities = thresholds, probabilities
        elif epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_weights)
    return records, best_epoch, best_thresholds, evaluate_week(best_probabilities >= best_thresholds, validation)


def run(args):
    sys.path.insert(0, str(args.upstream.resolve() / "examples"))
    from model import Model
    from text_embedder import GloveTextEmbedding

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(1 if device.type == "cuda" else 4)
    validation_time, holdout_time = map(hour_boundary, (args.validation_cutoff, args.holdout_cutoff))
    if validation_time + WEEK > holdout_time:
        raise ValueError("The validation week must finish before holdout.")
    users = pd.Index(np.loadtxt(args.cohort, dtype=np.int64, ndmin=1))
    if users.empty or not users.is_unique:
        raise ValueError("Expected a nonempty cohort of unique users.")
    coverage = Coverage.read(args.manifest, args.missing_activity_csv)
    data = ActivityData(args.db, coverage)
    pretrain_cutoffs = training_cutoffs(coverage, validation_time, 24, stride=DAY, target_horizon=DAY)
    gru_cutoffs = training_cutoffs(coverage, validation_time, 24, stride=DAY)
    if not pretrain_cutoffs or not gru_cutoffs:
        raise ValueError("The manifest does not contain sufficient historical training days.")
    print(f"Mixed pretraining on {device}: {len(args.variants)} variants, {args.samples:,} examples per epoch", flush=True)
    db, cold_id = read_database(args.db, holdout_time, coverage.start)
    graph, stats, _ = make_graph(db, GloveTextEmbedding, fit_before=validation_time)
    first_relationship = first_visible_relationship(graph).numpy()
    original_ids = pd.Series(data.user_codes.index, index=data.user_codes.to_numpy()).reindex(users, fill_value=-1).to_numpy()
    event_horizons = sorted({hours for hours, share in args.variants if share})
    events = event_samples(data, users, pretrain_cutoffs[0], validation_time, event_horizons) if event_horizons else {}
    snapshots = [data.snapshot(t, users, labeled=True, daily_targets=True) for t in gru_cutoffs]
    validation = data.snapshot(validation_time, users, labeled=True)
    holdout = data.snapshot(holdout_time, users, labeled=True)
    if not known_labels(validation).any() or not known_labels(holdout).any():
        raise ValueError("Both weekly evaluations require at least one fully observed activity.")
    train_loader, train_mask = make_loader(graph, data, users, snapshots, cold_id, args.batch_size, False, hops=HOPS)
    validation_loader, _ = make_loader(graph, data, users, [validation], cold_id, args.batch_size, False, hops=HOPS)
    holdout_loader, _ = make_loader(graph, data, users, [holdout], cold_id, args.batch_size, False, hops=HOPS)
    train_y = train_loader.transform.target.to(device)
    train_known = train_mask.to(device)
    if not train_mask.any():
        raise ValueError("The recurrent model requires known training labels.")
    del snapshots

    summary = {"seed": args.seed, "samples_per_epoch": args.samples, "pretrain_epochs": args.pretrain_epochs,
               "graph_batch_size": args.batch_size, "gru_batch_size": args.gru_batch_size,
               "contrastive_weight": args.contrastive_weight,
               "encoder_validation_samples": args.encoder_validation_samples,
               "pretraining_start_taipei": taipei_date(pretrain_cutoffs[0]),
               "validation_week": taipei_date(validation_time), "holdout_week": taipei_date(holdout_time),
               "gru_cutoffs": [taipei_date(cutoff) for cutoff in gru_cutoffs], "variants": []}
    for hours, share in args.variants:
        tag = f"h{hours}_event{share}"
        event_count = args.samples * share // 100
        calendar = calendar_samples(data, users, pretrain_cutoffs[0], validation_time, hours,
                                    args.samples - event_count, args.seed)
        samples = calendar
        if event_count:
            pool = events[hours]
            selected = np.random.default_rng(args.seed).permutation(len(pool.users))[:event_count]
            if len(selected) != event_count:
                raise ValueError(f"{tag} needs {event_count} distinct event-triggered samples.")
            samples = Samples(*(np.concatenate([fixed, event[selected]]) for fixed, event in zip(calendar, pool)))
        print(f"{tag}: {len(samples.users):,} samples, {event_count:,} event anchors, "
              f"{samples.targets.any(axis=1).mean():.2%} with activity", flush=True)

        seed_everything(args.seed)
        encoder = Model(data=graph, col_stats_dict=stats, num_layers=HOPS, channels=CHANNELS,
                        out_channels=len(EVENT_COLS), aggr="sum", norm="batch_norm", gnn="sage")
        history = RelBenchActivityHistory(encoder, cold_id).to(device)
        loader, mask = pretrain_loader(graph, samples, original_ids, first_relationship, cold_id, args.batch_size)
        if args.encoder_validation_samples:
            probe = calendar_samples(data, users, validation_time, validation_time + WEEK, hours,
                                     args.encoder_validation_samples, args.seed + 1)
            probe_loader, _ = pretrain_loader(graph, probe, original_ids, first_relationship, cold_id,
                                              args.batch_size, shuffle=False)
        initialize_loader_prior(history, loader, mask)
        optimizer = torch.optim.AdamW(history.parameters(), lr=DAILY_LEARNING_RATE)
        pretraining = []
        for epoch in range(1, args.pretrain_epochs + 1):
            started = perf_counter()
            if args.contrastive_weight:
                losses = train_contrastive_epoch(history, optimizer, loader, mask, device, args.contrastive_weight)
            else:
                loss, _ = train_epoch(history, optimizer, loader, mask, device)
                losses = {"loss": loss}
            record = {"epoch": epoch, **losses, "seconds": perf_counter() - started}
            if args.encoder_validation_samples and epoch in (3, args.pretrain_epochs):
                record["validation"] = encoder_validation(history, probe_loader, probe, device, args.seed)
            pretraining.append(record)
            print(json.dumps({"variant": tag, "stage": "pretrain", **record}), flush=True)

        history.freeze_encoder()
        seed_everything(args.seed)
        # The shared downstream head starts with next-day rather than short-window priors.
        history.classifier.reset_parameters()
        model = NextDayActivityModel(history, DAILY_HIDDEN_SIZE, len(EVENT_COLS)).to(device)
        initialize_loader_prior(model, train_loader, train_mask)
        train_vectors = cache_corrections(history, train_loader, device, args.seed)
        val_vectors = cache_corrections(history, validation_loader, device, args.seed)
        print(f"{tag}: training the GRU on {len(train_vectors):,} cached frozen graph corrections", flush=True)
        epochs, best_epoch, thresholds, val_metrics = fit_gru(model, train_vectors, train_y, train_known,
                                                             val_vectors, validation, args, device)
        test_vectors = cache_corrections(history, holdout_loader, device, args.seed)
        predictions = forecast_probabilities(model, test_vectors, args.gru_batch_size) >= thresholds
        result = {"name": tag, "horizon_hours": hours, "event_share_pct": share, "event_samples": event_count,
                  "active_samples": int(samples.targets.any(axis=1).sum()),
                  "sample_sha256": hashlib.sha256(b"".join(value.tobytes() for value in samples)).hexdigest(),
                  "users": int(np.unique(samples.users).size), "pretraining": pretraining,
                  "positive_counts": dict(zip(EVENT_COLS, samples.targets.sum(axis=0).tolist())),
                  "gru_epochs": epochs, "best_epoch": best_epoch, "validation": val_metrics,
                  "holdout": evaluate_week(predictions, holdout),
                  "thresholds": dict(zip(EVENT_COLS, thresholds.tolist()))}
        summary["variants"].append(result)
        path = args.output / f"{tag}.pt"
        write_atomically(path, lambda path: torch.save({"model": model.state_dict(), "column_stats": stats,
                                                       "result": result}, path))
        write_atomically(args.output / "metrics.json", lambda path: path.write_text(json.dumps(summary, indent=2) + "\n"))
        print(json.dumps({"variant": tag, "validation_score": val_metrics["score"],
                          "holdout_score": result["holdout"]["score"]}), flush=True)
        del train_vectors, val_vectors, test_vectors, optimizer, model, history, encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "data/processed/db")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/download_manifest.json")
    parser.add_argument("--cohort", type=Path, default=ROOT / "scripts/cohort.txt")
    parser.add_argument("--missing-activity-csv", type=Path)
    parser.add_argument("--upstream", type=Path, default=RUN_DIR / "upstream")
    parser.add_argument("--output", type=Path, default=RUN_DIR / "mixed_sampling")
    parser.add_argument("--validation-cutoff", default="2026-11-01")
    parser.add_argument("--holdout-cutoff", default="2026-11-08")
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--variant", dest="variants", action="append", nargs=2, type=int,
                        metavar=("HOURS", "EVENT_PERCENT"),
                        help="Forecast horizon and percentage of event-triggered samples. Repeat to compare variants.")
    parser.add_argument("--samples", type=int, default=36000)
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--contrastive-weight", type=float, default=0.0,
                        help="Weight on the active activity-profile contrastive loss.")
    parser.add_argument("--encoder-validation-samples", type=int, default=0,
                        help="Score short-window encoder forecasts on this many validation user-cutoffs.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gru-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.variants = [tuple(variant) for variant in args.variants] if args.variants else VARIANTS
    if any(hours < 1 or 24 % hours or not 0 <= share <= 100 for hours, share in args.variants):
        parser.error("Each variant needs a horizon that divides 24 hours and an event percent from 0 to 100.")
    if len(set(args.variants)) != len(args.variants):
        parser.error("Each variant must be distinct.")
    if min(args.samples, args.pretrain_epochs, args.epochs, args.patience, args.batch_size, args.gru_batch_size) < 1:
        parser.error("Sample counts, epoch counts, patience and batch sizes must be positive.")
    if (not np.isfinite(args.contrastive_weight) or args.contrastive_weight < 0 or
            args.encoder_validation_samples < 0):
        parser.error("Contrastive weight and encoder validation sample count must be finite and nonnegative.")
    run(args)


if __name__ == "__main__":
    main()

"""Train the ASG2 weekly graph model or write a submission from a checkpoint."""

import argparse
import copy
from functools import partial
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from evaluate import EVENT_COLS, OUTPUT_COLUMNS, USER_COL, f1, read_predictions
from gnn_model import DailyActivityModel, NextDayActivityModel, WeeklyActivityModel
from model_data import ActivityData, Coverage, DAY, WEEK, hour_boundary, taipei_date


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUTOFF = "2026-11-22T00:00+08:00"


def training_cutoffs(coverage, before, min_history_hours, *, stride=WEEK, target_horizon=WEEK):
    """Return cutoffs whose target windows end by the split boundary."""
    chosen = []
    cutoff = before - target_horizon
    while cutoff >= coverage.start:
        if cutoff + target_horizon <= coverage.end and coverage.has_history(cutoff, min_history_hours):
            chosen.append(cutoff)
        cutoff -= stride
    return list(reversed(chosen))


def training_labels(snapshot):
    if snapshot.next_day_counts is not None:
        return snapshot.next_day_counts, snapshot.next_day_mask
    return snapshot.target_counts, snapshot.label_mask


def known_labels(snapshot):
    # F1 needs observable negatives as well as positives. An entire activity is skipped
    # for a week with incomplete labels, as in the official evaluation's --skip option.
    return snapshot.label_mask.all(axis=0)


def confusion(prediction, truth):
    tp = (prediction & truth).sum(axis=0)
    fp = (prediction & ~truth).sum(axis=0)
    fn = (~prediction & truth).sum(axis=0)
    return np.stack([tp, fp, fn], axis=-1)


def activity_score(counts):
    macro = float(np.mean([f1(tp, fp, fn) for tp, fp, fn in counts]))
    micro = float(f1(*counts.sum(axis=0)))
    return macro, micro, (macro + micro) / 2


def evaluate_week(predictions, snapshot, users=None):
    activities = known_labels(snapshot)
    if users is None:
        users = np.ones(len(snapshot.target_counts), dtype=bool)
    if not users.any() or not activities.any():
        return None
    truth = snapshot.target_counts[users][:, activities] > 0
    counts = confusion(predictions[users][:, activities], truth)
    names = np.asarray(EVENT_COLS)[activities]
    macro, micro, score = activity_score(counts)
    return {"users": int(users.sum()), "score": score, "macro_f1": macro, "micro_f1": micro,
            "per_activity": {str(name): float(f1(*row)) for name, row in zip(names, counts)},
            "skipped_activities": np.asarray(EVENT_COLS)[~activities].tolist()}


def choose_thresholds(probabilities, snapshot):
    """Tune one threshold per activity with coordinate search on the assignment score."""
    activities = known_labels(snapshot)
    truth = snapshot.target_counts[:, activities] > 0
    probabilities = probabilities[:, activities]
    thresholds = np.full(len(EVENT_COLS), 0.5)
    active_thresholds = thresholds[activities].copy()
    counts = confusion(probabilities >= active_thresholds, truth)
    best_score = activity_score(counts)[2]

    candidates = []
    for column in probabilities.T:
        grid = np.quantile(column, np.linspace(0, 1, 41))
        candidates.append(np.unique(np.r_[grid, 0.5, 1.0]))

    for _ in range(2):
        for activity, grid in enumerate(candidates):
            if not truth[:, activity].any():
                continue
            column = probabilities[:, activity]
            options = confusion(column[:, None] >= grid, truth[:, activity, None])
            trial = counts.copy()
            for threshold, option in zip(grid, options):
                trial[activity] = option
                score = activity_score(trial)[2]
                if score > best_score + 1e-12:
                    best_score = score
                    active_thresholds[activity] = threshold
                    counts[activity] = option

    thresholds[activities] = active_thresholds
    return thresholds, best_score


def probabilities(model, snapshot):
    """Return scores for weekly decisions. A daily forecast uses OR at a shared threshold."""
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(snapshot))
        # max(score) >= threshold iff at least one daily score crosses it.
        if scores.ndim == 3:
            scores = scores.amax(dim=1)
        return scores.cpu().numpy()


def activity_model(sizes, hidden_size, *, daily_graphs, daily_loss):
    model_type = DailyActivityModel if daily_graphs else WeeklyActivityModel
    model = model_type(sizes, hidden_size, len(EVENT_COLS))
    if daily_loss:
        model = NextDayActivityModel(model, hidden_size, len(EVENT_COLS))
    return model


def graph_builder(data, *, daily_graphs, history_days):
    if daily_graphs:
        return data.daily_snapshot
    if history_days is not None:
        return partial(data.recent_snapshot, days=history_days)
    return data.snapshot


def make_model(training, hidden_size, device):
    sizes = {name: feature.shape[1] for name, feature in training[0].features.items()}
    model = activity_model(sizes, hidden_size, daily_graphs=bool(training[0].daily_graphs),
                           daily_loss=training[0].next_day_counts is not None)
    positives = np.zeros(len(EVENT_COLS), dtype=np.int64)
    known = np.zeros_like(positives)
    for snapshot in training:
        counts, mask = training_labels(snapshot)
        positives += ((counts > 0) & mask).reshape(-1, len(EVENT_COLS)).sum(axis=0)
        known += mask.reshape(-1, len(EVENT_COLS)).sum(axis=0)
    model.initialize_prior(positives, known)
    return model.to(device), sizes


def fit(training, args, device, validation=None, epochs=None):
    model, sizes = make_model(training, args.hidden_size, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    batches = []
    for snapshot in training:
        counts, mask = training_labels(snapshot)
        if mask.any():
            target = torch.as_tensor(counts > 0, dtype=torch.float32, device=device)
            batches.append((snapshot, target, torch.as_tensor(mask, device=device)))
    if not batches:
        raise ValueError("The training snapshots have no usable activity labels.")
    best = None
    max_epochs = args.epochs if epochs is None else epochs
    generator = np.random.default_rng(args.seed)
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for index in generator.permutation(len(batches)):
            snapshot, target, mask = batches[index]
            optimizer.zero_grad(set_to_none=True)
            logits = model(snapshot)
            loss = F.binary_cross_entropy_with_logits(logits[mask], target[mask])
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().item()
        average_loss = running_loss / len(batches)

        if validation is None:
            print(f"Refit epoch {epoch:02d}  loss={average_loss:.4f}", flush=True)
            continue

        validation_probabilities = probabilities(model, validation)
        thresholds, score = choose_thresholds(validation_probabilities, validation)
        print(f"Epoch {epoch:02d}  loss={average_loss:.4f}  validation_score={score:.4f}", flush=True)
        if best is None or score > best["score"] + 1e-12:
            best = {"epoch": epoch, "score": score, "thresholds": thresholds,
                    "weights": copy.deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= args.patience:
            break

    if best is not None:
        model.load_state_dict(best.pop("weights"))
    return model, sizes, best


def week_report(model, snapshot, thresholds):
    prediction = probabilities(model, snapshot) >= thresholds
    warm_users = snapshot.user_nodes.cpu().numpy() >= 0
    return {"start_taipei": taipei_date(snapshot.cutoff),
            "model": evaluate_week(prediction, snapshot),
            "copy_last_week": evaluate_week(snapshot.last_week, snapshot),
            "users_with_history": evaluate_week(prediction, snapshot, warm_users),
            "users_without_history": evaluate_week(prediction, snapshot, ~warm_users)}


def write_atomically(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f"{path.stem}-", suffix=path.suffix,
                                     delete=False) as file:
        temporary = Path(file.name)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_training(args, data, device):
    validation_cutoff = hour_boundary(args.validation_cutoff)
    holdout_cutoff = hour_boundary(args.holdout_cutoff)
    prediction_cutoff = hour_boundary(args.prediction_cutoff)
    if not (validation_cutoff + WEEK <= holdout_cutoff and holdout_cutoff + WEEK <= prediction_cutoff):
        raise ValueError("Validation, holdout and prediction cutoffs must be chronological and at least a week apart.")
    if (prediction_cutoff - validation_cutoff) % WEEK or (prediction_cutoff - holdout_cutoff) % WEEK:
        raise ValueError("Validation and holdout must share the prediction cutoff's weekly boundary.")

    user_ids = np.loadtxt(args.cohort, dtype=np.int64, ndmin=1)
    if len(user_ids) == 0 or len(np.unique(user_ids)) != len(user_ids):
        raise ValueError("The cohort must contain unique user IDs.")
    stride = DAY if args.daily_loss else WEEK
    cutoffs = training_cutoffs(data.coverage, validation_cutoff, args.min_history_hours, stride=stride)
    if not cutoffs:
        raise ValueError("There are no complete training weeks before the validation cutoff.")
    snapshot_builder = graph_builder(data, daily_graphs=args.daily_graphs, history_days=args.graph_history_days)
    build_snapshot = partial(snapshot_builder, daily_targets=args.daily_loss)
    training = [build_snapshot(cutoff, user_ids, labeled=True).to(device) for cutoff in cutoffs]
    validation = build_snapshot(validation_cutoff, user_ids, labeled=True).to(device)
    holdout = build_snapshot(holdout_cutoff, user_ids, labeled=True).to(device)
    if not known_labels(validation).any() or not known_labels(holdout).any():
        raise ValueError("Validation and holdout each need at least one activity with complete labels.")
    graph_mode = "daily" if args.daily_graphs else "cumulative"
    if args.graph_history_days is not None:
        graph_mode = f"recent_{args.graph_history_days}d"
    forecast_loss = "next_day" if args.daily_loss else "weekly"
    print(f"Device: {device}. Graph mode: {graph_mode}. Loss: {forecast_loss}. "
          f"Training cutoffs: {', '.join(map(taipei_date, cutoffs))}",
          flush=True)
    model, sizes, selected = fit(training, args, device, validation)
    thresholds = selected["thresholds"]

    training_key = "training_cutoffs_taipei" if args.daily_loss else "training_weeks_taipei"
    summary = {training_key: list(map(taipei_date, cutoffs)),
               "validation": week_report(model, validation, thresholds),
               "holdout": week_report(model, holdout, thresholds),
               "selected_epoch": selected["epoch"],
               "thresholds": dict(zip(EVENT_COLS, thresholds.tolist())),
               "threshold_default_activities": np.asarray(EVENT_COLS)[~known_labels(validation)].tolist(),
               "coverage_start_utc": data.coverage.start.isoformat(),
               "coverage_end_utc_exclusive": data.coverage.end.isoformat(),
               "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               "missing_activity_sha256": (
                   hashlib.sha256(args.missing_activity_csv.read_bytes()).hexdigest()
                   if args.missing_activity_csv is not None else None
               ),
               "seed": args.seed, "graph_mode": graph_mode, "graph_history_days": args.graph_history_days,
               "forecast_loss": forecast_loss,
               "min_history_hours": args.min_history_hours,
               "learning_rate": args.learning_rate}

    if args.refit:
        # Freeze model choices first. The refit can then include newly observed label weeks.
        snapshots = {item.cutoff: item for item in [*training, validation, holdout]}
        refit_cutoffs = training_cutoffs(data.coverage, prediction_cutoff, args.min_history_hours, stride=stride)
        final_training = [snapshots[t] if t in snapshots else build_snapshot(t, user_ids, labeled=True).to(device)
                          for t in refit_cutoffs]
        torch.manual_seed(args.seed)
        model, sizes, _ = fit(final_training, args, device, epochs=selected["epoch"])
        refit_key = "refit_cutoffs_taipei" if args.daily_loss else "refit_weeks_taipei"
        summary[refit_key] = list(map(taipei_date, refit_cutoffs))

    checkpoint = {"feature_sizes": sizes, "hidden_size": args.hidden_size, "events": EVENT_COLS,
                  "weights": model.state_dict(), "thresholds": thresholds.tolist(),
                  "prediction_cutoff": args.prediction_cutoff, "daily_graphs": args.daily_graphs,
                  "graph_history_days": args.graph_history_days,
                  "daily_loss": args.daily_loss,
                  "summary": summary}
    write_atomically(args.checkpoint, lambda path: torch.save(checkpoint, path))
    print(json.dumps(summary, indent=2))
    print(f"Saved checkpoint: {args.checkpoint}")


def run_prediction(args, data, device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["events"] != EVENT_COLS:
        raise ValueError("The checkpoint has different target activities.")
    cutoff = hour_boundary(args.prediction_cutoff or checkpoint["prediction_cutoff"])
    if data.coverage.end < cutoff:
        print(f"Input history ends {cutoff - data.coverage.end} before the prediction cutoff.", flush=True)

    user_ids = pd.read_csv(args.input, dtype={USER_COL: "int64"})[USER_COL]
    if user_ids.empty or user_ids.duplicated().any():
        raise ValueError("The prediction input must contain unique user IDs.")
    uses_daily_graphs = checkpoint.get("daily_graphs", False)
    build_snapshot = graph_builder(data, daily_graphs=uses_daily_graphs,
                                   history_days=checkpoint.get("graph_history_days"))
    snapshot = build_snapshot(cutoff, user_ids).to(device)
    actual_sizes = {name: values.shape[1] for name, values in snapshot.features.items()}
    if actual_sizes != checkpoint["feature_sizes"]:
        raise ValueError("The checkpoint was trained with a different feature layout.")
    model = activity_model(actual_sizes, checkpoint["hidden_size"], daily_graphs=uses_daily_graphs,
                           daily_loss=checkpoint.get("daily_loss", False))
    model.load_state_dict(checkpoint["weights"])
    model.to(device)
    prediction = (probabilities(model, snapshot) >= np.asarray(checkpoint["thresholds"])).astype(int)
    output = pd.DataFrame(prediction, columns=EVENT_COLS)
    output.insert(0, USER_COL, user_ids.to_numpy())

    def write_submission(path):
        output.to_csv(path, index=False, columns=OUTPUT_COLUMNS)
        read_predictions(path, expected_users=user_ids)

    write_atomically(args.pred, write_submission)
    cold = int((snapshot.user_nodes < 0).sum().item())
    print(f"Saved {len(output):,} predictions ({cold:,} using only the shared vector): {args.pred}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", type=Path, default=ROOT / "data/processed/db")
    common.add_argument("--manifest", type=Path, default=ROOT / "data/download_manifest.json")
    common.add_argument("--missing-activity-csv", type=Path,
                        help="Known source gaps, with columns activity,start,end (exclusive end).")
    common.add_argument("--checkpoint", type=Path,
                        help="Default: asg2.pt. Daily graphs and next-day loss use separate filenames.")
    common.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    common.add_argument("--threads", type=int, default=4, help="CPU threads used by PyTorch.")
    actions = parser.add_subparsers(dest="action", required=True)
    train = actions.add_parser("train", parents=[common], help="Train and score chronological weeks.")
    train.add_argument("--cohort", type=Path, default=ROOT / "scripts/cohort.txt")
    train.add_argument("--prediction-cutoff", default=DEFAULT_CUTOFF)
    train.add_argument("--validation-cutoff", default="2026-11-01")
    train.add_argument("--holdout-cutoff", default="2026-11-08")
    train.add_argument("--min-history-hours", type=int, default=24,
                       help="Required hours immediately before a training week. Longer windows carry coverage features.")
    train.add_argument("--hidden-size", type=int, default=64)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--patience", type=int, default=6)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--seed", type=int, default=42)
    history = train.add_mutually_exclusive_group()
    history.add_argument("--daily-graphs", action="store_true",
                         help="Use seven daily history graphs to predict the same weekly targets.")
    history.add_argument("--graph-history-days", type=int,
                         help="Start the input graph with recent activity rows and keep their referenced entities.")
    train.add_argument("--daily-loss", action="store_true",
                       help="Train seven next-day predictions; evaluate their OR on the original weekly targets.")
    train.add_argument("--refit", action="store_true", help="Refit after holdout on all eligible pre-test weeks.")
    predict = actions.add_parser("predict", parents=[common], help="Write a submission CSV.")
    predict.add_argument("--prediction-cutoff", help="Defaults to the cutoff stored in the checkpoint.")
    predict.add_argument("--input", type=Path, default=ROOT / "submission/test_input.csv")
    predict.add_argument("--pred", type=Path, required=True)
    args = parser.parse_args()

    if args.checkpoint is None:
        filename = "daily_asg2.pt" if args.action == "train" and args.daily_graphs else "asg2.pt"
        if args.action == "train" and args.graph_history_days is not None:
            filename = f"recent_{args.graph_history_days}d_{filename}"
        if args.action == "train" and args.daily_loss:
            filename = f"next_day_{filename}"
        args.checkpoint = ROOT / "data/model" / filename

    if args.threads < 1:
        parser.error("--threads must be positive.")
    if args.action == "train":
        sizes = (args.min_history_hours, args.hidden_size, args.epochs, args.patience, args.learning_rate)
        if any(value <= 0 for value in sizes):
            parser.error("Training sizes, epoch counts and learning rate must be positive.")
        if args.graph_history_days is not None and args.graph_history_days <= 0:
            parser.error("--graph-history-days must be positive.")
        torch.manual_seed(args.seed)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Try --device cpu.")
    torch.set_num_threads(args.threads)
    device = torch.device(device_name)
    try:
        coverage = Coverage.read(args.manifest, args.missing_activity_csv)
        data = ActivityData(args.db, coverage)
        if args.action == "train":
            run_training(args, data, device)
        else:
            run_prediction(args, data, device)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

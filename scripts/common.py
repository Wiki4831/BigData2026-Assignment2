"""Shared helpers: training cutoffs, labels, threshold search, scoring and atomic file writes."""

import tempfile
from pathlib import Path

import numpy as np

from evaluate import EVENT_COLS, f1
from model_data import WEEK

ROOT = Path(__file__).resolve().parents[1]


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


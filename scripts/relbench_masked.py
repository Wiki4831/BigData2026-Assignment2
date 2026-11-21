"""Pretrain on historical activity types with the target event rows hidden."""

import copy
import json
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from relbench.modeling.graph import AttachTargetTransform
from torch.nn import functional as F
from torch_geometric.loader import NeighborLoader

from evaluate import EVENT_COLS
from model_data import ACTIVITY_SOURCE, DAY, WEEK, taipei_date
from preprocess import TABLES


ENTITY = "Users"
NUM_TYPES = len(EVENT_COLS)
MASK_FOLDS = 8
LOCAL_OFFSET = pd.Timedelta(hours=8)
GROUP_COLUMNS = ["user_id", "cutoff", "activity"]


def masked_activity_events(data, user_ids, validation_end):
    """Use complete local days and group eligible events by first visibility."""
    events = data.events.loc[data.events["github_user_id"].isin(user_ids)].copy()
    visible = events[["event_time", "row_time"]].max(axis=1)
    events["cutoff"] = (visible + LOCAL_OFFSET).dt.floor("D") - LOCAL_OFFSET + DAY
    eligible = [cutoff for cutoff in events["cutoff"].unique()
                if cutoff <= validation_end and data.coverage.has_history(cutoff, DAY / pd.Timedelta(hours=1))]
    return events.loc[events["cutoff"].isin(eligible)]


def mask_activity_graph(graph, events, issues):
    """Isolate target rows before temporal sampling, in both edge directions."""
    hidden = {name: torch.zeros(graph[name].num_nodes, dtype=torch.bool) for name in graph.node_types}
    for index, activity in enumerate(EVENT_COLS):
        table, _, _ = ACTIVITY_SOURCE[activity]
        rows = events.loc[events["activity"].eq(index), "row_id"].to_numpy(dtype=np.int64)
        hidden[table][torch.from_numpy(rows)] = True

    # A PR conversation can expose the same opening through its mirrored issue row.
    pr_ids = hidden["PullRequests"].nonzero(as_tuple=True)[0].numpy()
    mirrors = issues.loc[issues["pull_request_id"].isin(pr_ids), TABLES["Issues"]["pk"]]
    hidden["Issues"][torch.from_numpy(mirrors.to_numpy(dtype=np.int64))] = True

    masked = copy.copy(graph)
    for source, relation, target in graph.edge_types:
        edge_type = (source, relation, target)
        edge_index = graph[edge_type].edge_index
        keep = ~(hidden[source][edge_index[0]] | hidden[target][edge_index[1]])
        masked[edge_type].edge_index = edge_index[:, keep]
    return masked


def first_visible_relationship(graph):
    """Find the first time a user's remaining event and user row can both be seen."""
    user_time = graph[ENTITY].time
    first = torch.full_like(user_time, torch.iinfo(user_time.dtype).max)
    for source, relation, target in graph.edge_types:
        if target == ENTITY:
            source_ids, user_ids = graph[source, relation, target].edge_index
            times = torch.maximum(graph[source].time[source_ids], user_time[user_ids])
            first.scatter_reduce_(0, user_ids, times, reduce="amin")
    return first


def build_masked_loaders(graph, events, issues, neighbors, batch_size, seed, *, shuffle):
    samples = events[GROUP_COLUMNS].drop_duplicates().copy()
    user_days = samples.groupby(["user_id", "cutoff"], sort=False)
    offsets = np.random.default_rng(seed).integers(MASK_FOLDS, size=user_days.ngroups)
    # Keep different activity targets for the same user-day in distinct graph views.
    samples["fold"] = (offsets[user_days.ngroup()] + user_days.cumcount()) % MASK_FOLDS
    rows = events.merge(samples, on=GROUP_COLUMNS, validate="many_to_one")
    eligible_counts = np.bincount(samples["activity"], minlength=NUM_TYPES)
    counts = np.zeros(NUM_TYPES, dtype=np.int64)
    loaders = []
    for fold, fold_rows in rows.groupby("fold", sort=True):
        masked = mask_activity_graph(graph, fold_rows, issues)
        group = samples.loc[samples["fold"].eq(fold)]
        nodes = torch.from_numpy(group["user_id"].to_numpy(dtype=np.int64))
        times = group["cutoff"].to_numpy(dtype="datetime64[s]").astype(np.int64) - 1
        times = torch.from_numpy(times)
        labels = torch.from_numpy(group["activity"].to_numpy(dtype=np.int64))
        # A seed needs some remaining history for the graph encoder to explain the target.
        known = first_visible_relationship(masked)[nodes] <= times
        if not known.any():
            continue
        labels = labels[known]
        counts += np.bincount(labels.numpy(), minlength=NUM_TYPES)
        loaders.append(NeighborLoader(masked, num_neighbors=neighbors, input_nodes=(ENTITY, nodes[known]),
                                      input_time=times[known], time_attr="time", temporal_strategy="uniform",
                                      transform=AttachTargetTransform(ENTITY, labels),
                                      batch_size=batch_size, shuffle=shuffle, num_workers=0))
    return loaders, eligible_counts, counts


def train_masked_epoch(model, loaders, weights, optimizer, device):
    model.train()
    total_loss, samples = 0.0, 0
    for index in torch.randperm(len(loaders)).tolist():
        for batch in loaders[index]:
            batch = batch.to(device)
            labels = batch[ENTITY].y
            optimizer.zero_grad(set_to_none=True)
            losses = F.cross_entropy(model(batch), labels, reduction="none")
            loss = (losses * weights[labels]).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            samples += len(labels)
    return total_loss / samples


@torch.no_grad()
def validate_masked(model, loaders, device, seed):
    model.eval()
    loss_by_type = torch.zeros(NUM_TYPES, dtype=torch.float64)
    confusion = torch.zeros((NUM_TYPES, NUM_TYPES), dtype=torch.long)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for loader in loaders:
            for batch in loader:
                batch = batch.to(device)
                labels = batch[ENTITY].y
                logits = model(batch)
                loss = F.cross_entropy(logits, labels, reduction="none").cpu().double()
                types = labels.cpu()
                predictions = logits.argmax(dim=1).cpu()
                loss_by_type.scatter_add_(0, types, loss)
                confusion += torch.bincount(types * NUM_TYPES + predictions,
                                            minlength=NUM_TYPES**2).reshape(NUM_TYPES, NUM_TYPES)
    totals = confusion.sum(dim=1)
    known = totals > 0
    balanced_loss = (loss_by_type[known] / totals[known]).mean().item()
    macro_recall = (confusion.diag()[known] / totals[known]).mean().item()
    return balanced_loss, macro_recall, confusion.tolist()


def pretrain_masked_activity(model, graph, data, user_ids, val_time, neighbors, args, device, learning_rate):
    events = masked_activity_events(data, user_ids, val_time + WEEK)
    issues = data.tables["Issues"].iloc[:graph["Issues"].num_nodes]
    train_events = events.loc[events["cutoff"].le(val_time)]
    val_events = events.loc[events["cutoff"].gt(val_time)]
    train_loaders, train_candidates, train_counts = build_masked_loaders(
        graph, train_events, issues, neighbors, args.batch_size, args.seed, shuffle=True)
    val_loaders, val_candidates, val_counts = build_masked_loaders(
        graph, val_events, issues, neighbors, args.batch_size, args.seed + 1, shuffle=False)
    if not train_loaders or not val_loaders:
        raise ValueError("Masked pretraining and validation both need samples with remaining activity history.")

    # Each type contributes equally even when one activity supplies more eligible targets.
    present = train_counts > 0
    class_weights = np.zeros(NUM_TYPES, dtype=np.float32)
    class_weights[present] = train_counts.sum() / (present.sum() * train_counts[present])
    weights = torch.from_numpy(class_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_loss = patience_loss = float("inf")
    last_improvement = 0
    records = []
    print(f"Masked activity type pretraining: {train_counts.sum():,}/{train_candidates.sum():,} train and "
          f"{val_counts.sum():,}/{val_candidates.sum():,} validation samples retain history", flush=True)
    for epoch in range(1, args.pretrain_epochs + 1):
        started = perf_counter()
        loss = train_masked_epoch(model, train_loaders, weights, optimizer, device)
        val_loss, val_recall, confusion = validate_masked(model, val_loaders, device, args.seed)
        if not np.isfinite(loss) or not np.isfinite(val_loss):
            raise ValueError(f"Non-finite masked pretraining loss at epoch {epoch}.")
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_weights = copy.deepcopy(model.state_dict())
            best_confusion = confusion
        if val_loss < patience_loss - args.pretrain_min_delta:
            patience_loss, last_improvement = val_loss, epoch
        record = {"epoch": epoch, "loss": loss, "validation_loss": val_loss,
                  "validation_macro_recall": val_recall, "seconds": perf_counter() - started}
        records.append(record)
        print(json.dumps({"stage": "masked_type_pretrain", **record}), flush=True)
        if epoch - last_improvement >= args.patience:
            break

    model.load_state_dict(best_weights)
    return {"objective": "masked_activity_type", "mask_folds": MASK_FOLDS,
            "feature_fit_before_taipei": taipei_date(val_time),
            "train_candidate_counts": dict(zip(EVENT_COLS, train_candidates.tolist())),
            "train_counts": dict(zip(EVENT_COLS, train_counts.tolist())),
            "validation_candidate_counts": dict(zip(EVENT_COLS, val_candidates.tolist())),
            "validation_counts": dict(zip(EVENT_COLS, val_counts.tolist())),
            "best_epoch": best_epoch, "best_validation_loss": best_loss,
            "best_validation_confusion": best_confusion, "epochs": records}

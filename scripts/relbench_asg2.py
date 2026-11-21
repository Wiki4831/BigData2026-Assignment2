"""Use the upstream RelBench entity GNN for weekly or recurrent daily ASG2 forecasts."""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from relbench.base import Database, Table, TaskType
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal, remove_pkey_fkey
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch_frame import stype
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_geometric import seed_everything
from torch_geometric.loader import NeighborLoader

from evaluate import EVENT_COLS
from gnn_model import NextDayActivityModel, initialize_activity_prior
from model_data import ActivityData, Coverage, DAY, WEEK, hour_boundary, taipei_date
from preprocess import TABLES
from relbench_masked import pretrain_masked_activity
from relbench_review_mask import pretrain_review_features
from train_model import (ROOT, choose_thresholds, evaluate_week, training_cutoffs, training_labels,
                         write_atomically)


RUN_DIR = ROOT / "data/model/relbench_public"
ENTITY = "Users"
TASK = SimpleNamespace(entity_table=ENTITY, entity_col="user_id", target_col="target",
                       task_type=TaskType.MULTILABEL_CLASSIFICATION)
DEFAULT_HOPS = 2
FIRST_HOP_NEIGHBORS = 128
CHANNELS = 128
LEARNING_RATE = 0.005
DAILY_HIDDEN_SIZE = 64
DAILY_LEARNING_RATE = 0.001

# Use attributes visible when their row first enters the graph. Codes for individual
# objects, mutable fields from later events, and post-close fields are omitted.
FEATURE_COLUMNS = {
    "Users": ("is_cohort",),
    "Repositories": (),
    "PullRequests": ("number", "title", "draft", "head_ref", "base_ref", "is_cross_repo", "created_at"),
    "Issues": ("number", "is_pull_request", "title", "body", "labels", "num_labels",
               "milestone_title", "issue_type", "locked", "created_at"),
    "IssueComments": ("is_pull_request", "body", "via_app"),
    "PRReviews": ("state", "body", "submitted_at"),
    "PRReviewComments": ("path", "file_extension", "subject_type", "position",
                         "original_position", "diff_hunk", "body"),
    "Pushes": ("ref_type", "branch", "is_new_ref"),
    "Owners": (),
    "Orgs": (),
}
CLIP_OLD_DAYS = 0
TEMPORAL_STRATEGY = "uniform"  # uniform | last (most recent neighbours first)
EXTRA_EDGES = "none"  # none | head | owner | both: PR.head_repo -> Repositories, Repositories.owner -> Users
SEQ_EDGES = False  # add "previous event of the same user in the same table" edges (event order on the graph)
FOCAL_GAMMA = 0.0  # >0 switches the daily BCE to focal loss
USER_COLUMN = {"PullRequests": "author_id", "Issues": "author_id", "IssueComments": "author_id",
               "PRReviews": "reviewer_id", "PRReviewComments": "author_id", "Pushes": "user_id"}
REPO_GROUP = "none"  # none | owner | org | both: extra Repositories -> group-table edges


class RelBenchActivityHistory(nn.Module):
    """Produce the initial GRU correction from the user's sampled history."""

    def __init__(self, graph_encoder, cold_id):
        super().__init__()
        self.graph_encoder = graph_encoder
        # The upstream model returns its seed embedding when the task head is identity.
        self.graph_encoder.head = nn.Identity()
        self.cold_id = cold_id
        self.projection = nn.Linear(CHANNELS, DAILY_HIDDEN_SIZE, bias=False)
        self.shared_user = nn.Parameter(torch.zeros(DAILY_HIDDEN_SIZE))
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(DAILY_HIDDEN_SIZE, len(EVENT_COLS))
        self.encoder_frozen = False
        self.extras = None  # optional per-user history module, trained already during pretraining

    def initialize_prior(self, counts, known):
        initialize_activity_prior(self.classifier, counts, known)

    def user_correction(self, batch):
        encoded = self.graph_encoder(batch, ENTITY)
        seed_ids = batch[ENTITY].n_id[:encoded.size(0)]
        # An isolated cold seed has no personal correction to the shared state.
        correction = self.projection(encoded)
        if self.extras is not None:
            correction = correction + self.extras(batch[ENTITY], encoded.size(0))
        return correction.masked_fill((seed_ids == self.cold_id)[:, None], 0)

    def initial_state(self, batch):
        return self.shared_user + self.dropout(self.user_correction(batch))

    def forward(self, batch):
        return self.classifier(self.initial_state(batch))

    def freeze_encoder(self):
        self.encoder_frozen = True
        for module in (self.graph_encoder, self.projection):
            module.requires_grad_(False)
            module.zero_grad(set_to_none=True)
            module.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.graph_encoder.eval()
            self.projection.eval()
        return self


def read_database(db_dir, cutoff, start):
    tables = {}
    for name, schema in TABLES.items():
        primary_key, row_time = schema["pk"], schema["time"]
        foreign_keys = {key: table for key, table in schema["fks"].items() if key != "merged_by_id"}
        columns = [primary_key, row_time, *foreign_keys, *FEATURE_COLUMNS[name]]
        if name == "Repositories" and REPO_GROUP != "none":
            columns += ["owner_login", "org_id"]
        if name == "PullRequests" and EXTRA_EDGES in ("head", "both"):
            columns += ["head_repo_id"]
        if name == "Repositories" and EXTRA_EDGES in ("owner", "both"):
            columns += ["owner_login"]
        rows = pd.read_parquet(db_dir / f"{name}.parquet", columns=columns)
        # Anonymised codes shared across tables give two extra relations (code -> primary key).
        if name == "PullRequests" and EXTRA_EDGES in ("head", "both"):
            repo_ids = pd.read_parquet(db_dir / "Repositories.parquet", columns=["repo_id", "github_repo_id"])
            rows["head_repo_pk"] = rows["head_repo_id"].map(repo_ids.set_index("github_repo_id")["repo_id"]).astype("Int64")
            rows = rows.drop(columns=["head_repo_id"])
            foreign_keys = {**foreign_keys, "head_repo_pk": "Repositories"}
        if name == "Repositories" and EXTRA_EDGES in ("owner", "both"):
            logins = pd.read_parquet(db_dir / "Users.parquet", columns=["user_id", "login"])
            rows["owner_user_pk"] = rows["owner_login"].map(logins.set_index("login")["user_id"]).astype("Int64")
            if REPO_GROUP == "none":
                rows = rows.drop(columns=["owner_login"])
            foreign_keys = {**foreign_keys, "owner_user_pk": "Users"}
        rows = rows.loc[rows[row_time].lt(cutoff)].reset_index(drop=True)
        rows[row_time] = rows[row_time].astype("datetime64[ns]")
        if SEQ_EDGES and name in USER_COLUMN:
            ordered = rows.sort_values([USER_COLUMN[name], row_time, primary_key], kind="stable")
            previous = ordered.groupby(USER_COLUMN[name], sort=False)[primary_key].shift(1)
            rows["prev_event_pk"] = previous.reindex(rows.index).astype("Int64")
            foreign_keys = {**foreign_keys, "prev_event_pk": name}
        if CLIP_OLD_DAYS:
            # Very old object timestamps fall below the year range fitted on training rows.
            # Floor at 1 January of the data start year, so no year falls below the fitted range.
            floor = pd.Timestamp(year=start.year, month=1, day=1)
            for column in FEATURE_COLUMNS[name]:
                if column.endswith("_at"):
                    rows[column] = rows[column].clip(lower=floor)
        if name == "PullRequests":
            # The static public graph cannot represent an author edge added later.
            delayed = rows["created_at"].gt(rows[row_time])
            rows.loc[delayed, ["author_id", "created_at"]] = None
        if name == "Repositories" and REPO_GROUP != "none":
            extra = {}
            for group, column, table_name in (("owner", "owner_login", "Owners"), ("org", "org_id", "Orgs")):
                if REPO_GROUP not in (group, "both"):
                    continue
                # Ids follow first appearance, so any earlier cutoff sees a prefix of the same ids.
                codes = pd.Series(pd.factorize(rows[column], sort=False)[0], index=rows.index)
                codes = codes.where(rows[column].notna())
                key = f"{group}_group_id"
                rows[key] = codes.astype("Int64")
                first = rows.dropna(subset=[key]).groupby(key)[row_time].min()
                extra[table_name] = Table(pd.DataFrame({f"{group}_group_id": first.index.astype("int64"),
                                                        row_time: first.to_numpy()}),
                                          {}, f"{group}_group_id", row_time)
                foreign_keys[key] = table_name
            rows = rows.drop(columns=[c for c in ("owner_login", "org_id") if c in rows])
            tables[name] = Table(rows, foreign_keys, primary_key, row_time)
            tables.update(extra)
            continue
        tables[name] = Table(rows, foreign_keys, primary_key, row_time)

    for table in tables.values():
        for key, target in table.fkey_col_to_pkey_table.items():
            table.df[key] = table.df[key].where(table.df[key].lt(len(tables[target].df)).fillna(False))

    # Cold users share an isolated seed, with no edges or user ID features.
    users = tables[ENTITY]
    cold_id = len(users.df)
    users.df.loc[cold_id] = [cold_id, start - pd.Timedelta(seconds=1), True]
    return Database(tables), cold_id


def make_graph(database, text_embedder, *, fit_before=None):
    fit_database = database
    if fit_before is not None:
        fit_database = Database({name: table.upto(fit_before - pd.Timedelta(seconds=1))
                                 for name, table in database.table_dict.items()})
    semantic_types = get_stype_proposal(fit_database)
    text_columns = []
    for name, table in fit_database.table_dict.items():
        types = semantic_types[name]
        for column in FEATURE_COLUMNS[name]:
            if not table.df[column].notna().any():
                types.pop(column, None)
            elif types[column] == stype.text_embedded:
                text_columns.append(f"{name}.{column}")

    print(f"Inferred text feature columns: {', '.join(text_columns)}", flush=True)
    config = None
    if text_columns:
        config = TextEmbedderConfig(text_embedder=text_embedder(device="cpu"), batch_size=256)
    if fit_before is None:
        graph, column_stats = make_pkey_fkey_graph(database, semantic_types, text_embedder_cfg=config)
        return graph, column_stats, text_columns

    # Use RelBench's PK/FK topology and fit the feature transforms on training rows.
    # The same fitted converters handle new rows at validation and holdout time.
    graph, _ = make_pkey_fkey_graph(database, {name: {} for name in semantic_types})
    column_stats = {}
    for name, table in fit_database.table_dict.items():
        types = semantic_types[name]
        remove_pkey_fkey(types, table)
        if table.df.empty or not types:
            constant = Dataset(df=pd.DataFrame({"__const__": [1.0]}),
                               col_to_stype={"__const__": stype.numerical}).materialize()
            column_stats[name] = constant.col_stats
            continue
        fitted = Dataset(df=table.df, col_to_stype=types, col_to_text_embedder_cfg=config).materialize()
        graph[name].tf = fitted.convert_to_tensor_frame(database.table_dict[name].df)
        column_stats[name] = fitted.col_stats
    return graph, column_stats, text_columns


def neighbors_by_hop(hops):
    return [max(1, FIRST_HOP_NEIGHBORS // (2 ** index)) for index in range(hops)]


class AttachHistory:
    """Attach each seed's precomputed history inputs to a sampled batch."""

    def __init__(self, inner, stats, sequence):
        self.inner, self.stats, self.sequence = inner, stats, sequence
        self.target = inner.target

    def __call__(self, batch):
        batch = self.inner(batch)
        index = batch[ENTITY].input_id
        if self.stats is not None:
            batch[ENTITY].hist_stats = self.stats[index]
        if self.sequence is not None:
            batch[ENTITY].hist_seq = self.sequence[index]
        return batch


HISTORY_MODE = "none"  # none | stats | seq | both
HISTORY_INPUTS = None
EDGE_DROP = 0.0   # probability of dropping any sampled edge during training
EVENT_MASK = 0.0  # extra drop probability for edges touching the seed entity (masks the user's own events)


class DropEdges:
    """Training-time augmentation: randomly remove sampled edges (DropEdge / event masking)."""

    def __init__(self, inner, edge_drop, event_mask):
        self.inner, self.edge_drop, self.event_mask = inner, edge_drop, event_mask
        self.target = inner.target

    def __call__(self, batch):
        batch = self.inner(batch)
        for edge_type in batch.edge_types:
            store = batch[edge_type]
            count = store.edge_index.size(1)
            if count == 0:
                continue
            p = self.edge_drop
            if ENTITY in (edge_type[0], edge_type[2]):
                p = 1 - (1 - p) * (1 - self.event_mask)
            if p <= 0:
                continue
            keep = torch.rand(count) >= p
            store.edge_index = store.edge_index[:, keep]
            if "e_id" in store:
                store.e_id = store.e_id[keep]
        return batch


def make_loader(graph, activity_data, user_ids, snapshots, cold_id, batch_size, shuffle, *, hops=DEFAULT_HOPS,
                augment=False):
    codes = activity_data.user_codes
    original_ids = pd.Series(codes.index, index=codes.to_numpy()).reindex(user_ids)
    rows = []
    masks = []
    for snapshot in snapshots:
        nodes = original_ids.where(snapshot.user_nodes.numpy() >= 0, cold_id).to_numpy(dtype=np.int64)
        counts, mask = training_labels(snapshot)
        # Public temporal sampling is inclusive. The ASG2 cutoff is exclusive.
        rows.append(pd.DataFrame({TASK.entity_col: nodes,
                                  "time": snapshot.cutoff - pd.Timedelta(seconds=1),
                                  TASK.target_col: list((counts > 0).astype(np.float32))}))
        masks.append(mask)
    targets = pd.concat(rows, ignore_index=True)
    # RelBench converts datetime integers from nanoseconds to seconds.
    targets["time"] = targets["time"].astype("datetime64[ns]")
    table = Table(targets, {TASK.entity_col: ENTITY}, time_col="time")
    inputs = get_node_train_table_input(table, TASK)
    transform = inputs.transform
    if HISTORY_MODE != "none" and HISTORY_INPUTS is not None:
        stats = (torch.from_numpy(np.concatenate([HISTORY_INPUTS.stats(s.cutoff) for s in snapshots]))
                 if HISTORY_MODE in ("stats", "both") else None)
        sequence = (torch.from_numpy(np.concatenate([HISTORY_INPUTS.sequence(s.cutoff) for s in snapshots]))
                    if HISTORY_MODE in ("seq", "both") else None)
        transform = AttachHistory(transform, stats, sequence)
    if augment and (EDGE_DROP > 0 or EVENT_MASK > 0):
        transform = DropEdges(transform, EDGE_DROP, EVENT_MASK)
    loader = NeighborLoader(graph, num_neighbors=neighbors_by_hop(hops), time_attr="time",
                            input_nodes=inputs.nodes, input_time=inputs.time,
                            transform=transform, temporal_strategy=TEMPORAL_STRATEGY,
                            batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader, torch.from_numpy(np.concatenate(masks)).bool()


def batch_logits(model, batch):
    if isinstance(model, (NextDayActivityModel, RelBenchActivityHistory)):
        return model(batch)
    return model(batch, ENTITY)


def hard_negative_ranking_loss(logits, labels, known, hard_ratio=3, min_hard=8):
    """Per activity, push each weekly positive above the highest-scoring weekly negatives in the batch."""
    if logits.ndim != 3:
        return logits.sum() * 0
    score = logits.amax(dim=1)
    positive = labels.amax(dim=1) > 0
    usable = known.all(dim=1)
    losses = []
    for activity in range(score.size(1)):
        pos = score[positive[:, activity] & usable[:, activity], activity]
        neg = score[~positive[:, activity] & usable[:, activity], activity]
        if not len(pos) or not len(neg):
            continue
        keep = min(len(neg), max(min_hard, hard_ratio * len(pos)))
        hard = neg.topk(keep).values
        losses.append(F.softplus(hard[None, :] - pos[:, None]).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0


def activity_loss(logits, labels):
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if FOCAL_GAMMA > 0:
        p_wrong = torch.where(labels > 0.5, 1 - torch.sigmoid(logits), torch.sigmoid(logits))
        loss = loss * p_wrong.detach() ** FOCAL_GAMMA
    return loss.mean()


def train_epoch(model, optimizer, loader, mask, device, audit_parameters=None, rank_weight=0.0):
    model.train()
    total_loss = 0.0
    label_count = 0
    first_update = {}
    for batch in loader:
        known = mask[batch[ENTITY].input_id]
        count = int(known.sum())
        if count == 0:
            continue
        batch = batch.to(device)
        known = known.to(device)
        optimizer.zero_grad()
        logits = batch_logits(model, batch)
        loss = activity_loss(logits[known], batch[ENTITY].y[known])
        if rank_weight:
            loss = loss + rank_weight * hard_negative_ranking_loss(logits, batch[ENTITY].y, known)
        loss.backward()
        if audit_parameters and not first_update:
            # Record the effect of this daily loss on actual graph and GRU weights.
            before = {name: param.detach().clone() for name, param in audit_parameters.items()}
            optimizer.step()
            for name, param in audit_parameters.items():
                grad = param.grad
                first_update[name] = {
                    "gradient_l2": 0.0 if grad is None else grad.norm().item(),
                    "weight_change_l2": (param.detach() - before[name]).norm().item(),
                }
        else:
            optimizer.step()
        total_loss += loss.item() * count
        label_count += count
    if not label_count:
        raise ValueError("The training data has no usable activity labels.")
    return total_loss / label_count, first_update


@torch.no_grad()
def predict(model, loader, device, daily_out=None):
    model.eval()
    probabilities = np.full((len(loader.input_data.node), len(EVENT_COLS)), np.nan, dtype=np.float32)
    for batch in loader:
        indices = batch[ENTITY].input_id.numpy()
        scores = batch_logits(model, batch.to(device)).sigmoid()
        if scores.ndim == 3:
            if daily_out is not None:
                if "x" not in daily_out:
                    daily_out["x"] = np.full((len(probabilities),) + tuple(scores.shape[1:]), np.nan, dtype=np.float32)
                daily_out["x"][indices] = scores.cpu().numpy()
            # A daily probability crossing the threshold is a weekly positive.
            scores = scores.amax(dim=1)
        probabilities[indices] = scores.cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("The public loader must supply a finite prediction for every target row.")
    return probabilities


def weekly_auc(probabilities, snapshot):
    scores = []
    for column in range(len(EVENT_COLS)):
        known = snapshot.label_mask[:, column]
        target = snapshot.target_counts[known, column] > 0
        if target.any() and not target.all():
            scores.append(roc_auc_score(target, probabilities[known, column]))
    if not scores:
        raise ValueError("No validation activity has both positive and negative labels.")
    return float(np.mean(scores))


def initialize_loader_prior(model, loader, mask):
    known = mask.reshape(-1, len(EVENT_COLS))
    targets = loader.transform.target.bool().reshape_as(known)
    model.initialize_prior((targets & known).sum(dim=0).numpy(), known.sum(dim=0).numpy())


@torch.no_grad()
def next_day_validation_bce(model, loader, mask, device, seed):
    model.eval()
    total_loss = 0.0
    label_count = int(mask.sum())
    if not label_count:
        raise ValueError("The validation dates have no usable activity labels.")
    # Repeat the same temporal samples each epoch without reseeding training.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for batch in loader:
            known = mask[batch[ENTITY].input_id].to(device)
            batch = batch.to(device)
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits[known], batch[ENTITY].y[known], reduction="sum")
            total_loss += loss.item()
    return total_loss / label_count


def pretrain_history(history, args, graph, data, user_ids, val_time, device, audit_parameters):
    cutoffs = training_cutoffs(data.coverage, val_time, args.min_history_hours,
                               stride=DAY, target_horizon=DAY)
    if not cutoffs:
        raise ValueError("There are no next-day targets before the validation cutoff.")
    snapshots = [data.snapshot(cutoff, user_ids, labeled=True, target_horizon=DAY) for cutoff in cutoffs]
    loader, mask = make_loader(graph, data, user_ids, snapshots, history.cold_id, args.batch_size, True,
                               hops=args.hops, augment=True)
    del snapshots
    initialize_loader_prior(history, loader, mask)
    optimizer = torch.optim.AdamW(history.parameters(), lr=DAILY_LEARNING_RATE)
    if args.pretrain_only:
        validation_cutoffs = pd.date_range(val_time, val_time + WEEK, freq=DAY, inclusive="left")
        validation = [data.snapshot(cutoff, user_ids, labeled=True, target_horizon=DAY)
                      for cutoff in validation_cutoffs]
        val_loader, val_mask = make_loader(graph, data, user_ids, validation, history.cold_id,
                                           args.batch_size, False, hops=args.hops)
        del validation
        best_loss = patience_loss = float("inf")
        last_improvement_epoch = 0
        stop_reason = "epoch_limit"

    records = []
    first_update = {}
    print(f"Pretraining the graph encoder for up to {args.pretrain_epochs} epochs on {len(cutoffs)} daily cutoffs",
          flush=True)
    for epoch in range(1, args.pretrain_epochs + 1):
        started = perf_counter()
        loss, update = train_epoch(history, optimizer, loader, mask, device,
                                   audit_parameters if epoch == 1 else None)
        if update:
            first_update = update
        if args.pretrain_only:
            val_loss = next_day_validation_bce(history, val_loader, val_mask, device, args.seed)
            if not np.isfinite(val_loss):
                raise ValueError(f"Non-finite validation loss at pretraining epoch {epoch}.")
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                best_weights = copy.deepcopy(history.state_dict())
            if val_loss < patience_loss - args.pretrain_min_delta:
                patience_loss, last_improvement_epoch = val_loss, epoch
        record = {"epoch": epoch, "loss": loss, "seconds": perf_counter() - started}
        if args.pretrain_only:
            record["validation_loss"] = val_loss
        records.append(record)
        print(json.dumps({"stage": "pretrain", **record}), flush=True)
        if args.pretrain_only and epoch - last_improvement_epoch >= args.patience:
            stop_reason = "validation_plateau"
            break

    report = {"cutoffs": [taipei_date(cutoff) for cutoff in cutoffs],
              "last_target_end_taipei": taipei_date(cutoffs[-1] + DAY),
              "feature_fit_before_taipei": taipei_date(val_time),
              "user_cutoffs": len(mask), "epochs": records, "first_optimizer_step": first_update}
    if args.pretrain_only:
        history.load_state_dict(best_weights)
        report.update(validation_cutoffs=[taipei_date(cutoff) for cutoff in validation_cutoffs],
                      validation_user_cutoffs=len(val_mask), best_epoch=best_epoch,
                      best_validation_loss=best_loss, last_improvement_epoch=last_improvement_epoch,
                      patience=args.patience, min_delta=args.pretrain_min_delta, stop_reason=stop_reason)
    return report


def load_encoder_checkpoint(path, history, column_stats, val_time):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    pretraining = checkpoint["pretraining"]
    if pretraining["feature_fit_before_taipei"] != taipei_date(val_time):
        raise ValueError("The encoder checkpoint and validation cutoff must use the same feature fitting window.")
    saved_stats = checkpoint["column_stats"]
    if saved_stats.keys() != column_stats.keys():
        raise ValueError("The encoder checkpoint uses different tables.")
    for table, columns in column_stats.items():
        if columns.keys() != saved_stats[table].keys():
            raise ValueError(f"The encoder checkpoint uses different columns in {table}.")
        for column, stats in columns.items():
            saved = saved_stats[table][column]
            if saved.keys() != stats.keys():
                raise ValueError(f"The encoder checkpoint uses different feature types in {table}.{column}.")
            for key, value in stats.items():
                matches = torch.equal(value, saved[key]) if torch.is_tensor(value) else value == saved[key]
                if not matches:
                    raise ValueError(f"The encoder checkpoint has different feature statistics in {table}.{column}.")
    history.load_state_dict(checkpoint["history"], strict=True)
    return pretraining


def run(args):
    upstream = args.upstream.resolve()
    sys.path.insert(0, str(upstream / "examples"))
    from model import Model
    from text_embedder import GloveTextEmbedding

    revision = subprocess.check_output(["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(args.threads or (1 if device.type == "cuda" else 4))
    seed_everything(args.seed)
    global SEQ_EDGES, FOCAL_GAMMA, EDGE_DROP, EVENT_MASK, REPO_GROUP, CLIP_OLD_DAYS, TEMPORAL_STRATEGY, EXTRA_EDGES, HISTORY_MODE, HISTORY_INPUTS
    HISTORY_MODE = args.history
    EDGE_DROP, EVENT_MASK = args.edge_drop, args.event_mask
    SEQ_EDGES, FOCAL_GAMMA = args.seq_edges, args.focal_gamma
    TEMPORAL_STRATEGY = args.temporal_strategy
    EXTRA_EDGES = args.extra_edges
    REPO_GROUP = args.repo_group
    CLIP_OLD_DAYS = args.clip_old_days
    val_time = hour_boundary(args.validation_cutoff)
    if args.pretrain_only:
        graph_cutoff = val_time + WEEK
    else:
        test_time = hour_boundary(args.holdout_cutoff)
        if val_time + WEEK > test_time:
            raise ValueError("The validation target must end before the holdout prediction starts.")
        graph_cutoff = test_time
    user_ids = np.loadtxt(args.cohort, dtype=np.int64, ndmin=1)
    if not len(user_ids) or len(np.unique(user_ids)) != len(user_ids):
        raise ValueError("The cohort must contain unique user IDs.")
    coverage = Coverage.read(args.manifest, args.missing_activity_csv)
    data = ActivityData(args.db, coverage)
    if args.train_gap_csv:
        data.train_gaps = Coverage.read(args.manifest, args.train_gap_csv)
        data.train_gap_min_fraction = args.train_gap_min_fraction
    if HISTORY_MODE != "none":
        from user_history import HistoryInputs
        if args.history_v2:
            users_table = pd.read_parquet(args.db / "Users.parquet", columns=["github_user_id", "first_seen_at"])
            HISTORY_INPUTS = HistoryInputs(data.events, user_ids, users_table,
                                           Coverage.read(args.manifest, args.source_gap_csv))
        else:
            HISTORY_INPUTS = HistoryInputs(data.events, user_ids)
    forecast = "next_day_gru" if args.daily_loss else "weekly"
    mode = "pretrain_only" if args.pretrain_only else forecast
    print(f"RelBench {revision[:12]} | {device} | {args.hops} hops | {len(user_ids):,} users | {mode}",
          flush=True)
    database, cold_id = read_database(args.db, graph_cutoff, coverage.start)
    graph, column_stats, text_columns = make_graph(database, GloveTextEmbedding,
                                                   fit_before=val_time if (args.pretrain_epochs or
                                                                           args.encoder_checkpoint) else None)
    print(f"{sum(graph.num_nodes_dict.values()):,} nodes | {graph.num_edges:,} directed edges | "
          f"{len(graph.edge_types)} edge types", flush=True)
    if not args.pretrain_only:
        stride = DAY if args.daily_loss else WEEK
        cutoffs = training_cutoffs(coverage, val_time, args.min_history_hours, stride=stride)
        if not cutoffs:
            raise ValueError("The manifest has no eligible training weeks.")
        training = [data.snapshot(cutoff, user_ids, labeled=True, daily_targets=args.daily_loss)
                    for cutoff in cutoffs]
        validation = data.snapshot(val_time, user_ids, labeled=True)
        holdout = data.snapshot(test_time, user_ids, labeled=True)
        print(f"Training on {len(training)} cutoffs for up to {args.epochs} epochs", flush=True)
        train_loader, train_mask = make_loader(graph, data, user_ids, training, cold_id, args.batch_size, True,
                                               hops=args.hops, augment=args.aug_gru)
        val_loader, _ = make_loader(graph, data, user_ids, [validation], cold_id, args.batch_size, False,
                                    hops=args.hops)
    if args.backbone == "relgnn":
        # RelGNN (Chen et al., ICML 2025): composite message passing over atomic routes.
        sys.path.insert(0, str(RUN_DIR / "relgnn_upstream/examples"))
        from atomic_routes import get_atomic_routes
        from relgnn_model import RelGNN_Model
        graph_model = RelGNN_Model(data=graph, col_stats_dict=column_stats, num_model_layers=args.hops,
                                   channels=CHANNELS, out_channels=len(EVENT_COLS), aggr="sum",
                                   norm="batch_norm", atomic_routes=get_atomic_routes(graph.edge_types),
                                   num_heads=args.relgnn_heads, simplified_MP=True)
    else:
        graph_model = Model(data=graph, col_stats_dict=column_stats, num_layers=args.hops,
                            channels=CHANNELS, out_channels=len(EVENT_COLS), aggr="sum",
                            norm="batch_norm", gnn="sage")
    if args.no_time_encoding:
        graph_model.temporal_encoder.forward = lambda *a, **k: {}
    audit_parameters = None
    pretraining = None
    if args.daily_loss or args.pretrain_only:
        history = RelBenchActivityHistory(graph_model, cold_id).to(device)
        if args.history_pretrain and HISTORY_MODE != "none":
            from user_history import HistoryExtras
            history.extras = HistoryExtras(HISTORY_MODE, DAILY_HIDDEN_SIZE, cold_id, args.history_v2).to(device)
        learning_rate = DAILY_LEARNING_RATE
        relation = ("Pushes", "f2p_user_id", ENTITY)
        audit_parameters = {} if args.backbone == "relgnn" else {
            f"push_to_user_layer_{index}": layer.convs[relation].lin_l.weight
            for index, layer in enumerate(graph_model.gnn.convs, 1)}
        if args.encoder_checkpoint:
            pretraining = load_encoder_checkpoint(args.encoder_checkpoint, history, column_stats, val_time)
            audit_parameters.update(shared_user=history.shared_user, projection=history.projection.weight)
            print(f"Loaded encoder checkpoint {args.encoder_checkpoint}", flush=True)
        elif args.pretrain_epochs:
            audit_parameters.update(shared_user=history.shared_user, projection=history.projection.weight)
            pretext_weights = None
            if args.masked_type_pretrain:
                pretraining = pretrain_masked_activity(history, graph, data, user_ids, val_time,
                                                       neighbors_by_hop(args.hops), args, device, learning_rate)
            elif args.review_feature_pretrain:
                pretraining, pretext_weights = pretrain_review_features(
                    history, graph, data, user_ids, val_time, neighbors_by_hop(args.hops), args, device,
                    learning_rate)
            else:
                pretraining = pretrain_history(history, args, graph, data, user_ids, val_time, device,
                                               audit_parameters)
            encoder_path = args.metrics.with_suffix(".encoder.pt")
            encoder_artifact = {"history": history.state_dict(), "column_stats": column_stats,
                                "pretraining": pretraining}
            if pretext_weights is not None:
                encoder_artifact["review_reconstruction"] = pretext_weights
            write_atomically(encoder_path, lambda path: torch.save(encoder_artifact, path))
            if args.pretrain_only:
                results = {"mode": mode, "relbench_commit": revision, "seed": args.seed,
                           "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                           "hops": args.hops, "neighbors": neighbors_by_hop(args.hops),
                           "batch_size": args.batch_size, "learning_rate": learning_rate,
                           "pretraining": pretraining}
                write_atomically(args.metrics, lambda path: path.write_text(json.dumps(results, indent=2) + "\n"))
                print(f"Pretraining finished at epoch {len(pretraining['epochs'])}: "
                      f"best validation BCE {pretraining['best_validation_loss']:.6f} "
                      f"at epoch {pretraining['best_epoch']} ({pretraining['stop_reason']})", flush=True)
                return

        pretraining_objective = pretraining.get("objective") if pretraining is not None else None
        if pretraining is not None:
            if args.encoder_finetune_lr is None:
                history.freeze_encoder()
                print("Training the GRU and shared forecast head with the graph encoder frozen", flush=True)
            else:
                print(f"Fine-tuning the graph encoder and projection at {args.encoder_finetune_lr:g}", flush=True)
            if pretraining_objective == "masked_activity_type":
                # A historical type classifier and a future occurrence head have different targets.
                history.classifier.reset_parameters()

        model = NextDayActivityModel(history, DAILY_HIDDEN_SIZE, len(EVENT_COLS)).to(device)
        model.recurrent = not args.no_recurrence
        if HISTORY_MODE != "none" and not args.history_pretrain:
            from user_history import HistoryExtras
            model.extras = HistoryExtras(HISTORY_MODE, DAILY_HIDDEN_SIZE, cold_id, args.history_v2).to(device)
        if pretraining is None or pretraining_objective in {"masked_activity_type", "masked_review_state"}:
            initialize_loader_prior(model, train_loader, train_mask)
        if args.encoder_finetune_lr is None:
            optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                          lr=learning_rate)
        else:
            encoder_parameters = list(history.graph_encoder.parameters()) + list(history.projection.parameters())
            encoder_ids = {id(parameter) for parameter in encoder_parameters}
            recurrent_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
            optimizer = torch.optim.AdamW([{"params": recurrent_parameters},
                                           {"params": encoder_parameters, "lr": args.encoder_finetune_lr}],
                                          lr=learning_rate)
        if args.encoder_checkpoint:
            initial_state = hashlib.sha256()
            for name, tensor in model.state_dict().items():
                initial_state.update(name.encode())
                initial_state.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            initial_state_hash = initial_state.hexdigest()
        audit_parameters["gru_input_weights"] = model.update_state.weight_ih
    else:
        model = graph_model.to(device)
        learning_rate = LEARNING_RATE
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_metric = float("-inf")
    epochs = []
    gradient_audit = {}
    for epoch in range(1, args.epochs + 1):
        started = perf_counter()
        loss, update = train_epoch(model, optimizer, train_loader, train_mask, device,
                                   audit_parameters if epoch == 1 else None, args.rank_weight)
        if update:
            gradient_audit = update
            print(json.dumps({"first_optimizer_step": update}), flush=True)
        val_probabilities = predict(model, val_loader, device)
        auc = weekly_auc(val_probabilities, validation)
        report = {"epoch": epoch, "loss": loss, "validation_macro_roc_auc": auc,
                  "seconds": perf_counter() - started}
        if args.daily_loss:
            _, metric = choose_thresholds(val_probabilities, validation)
            report["validation_score"] = metric
        else:
            metric = auc
        epochs.append(report)
        print(json.dumps(report), flush=True)
        if metric > best_metric + 1e-12:
            best_metric = metric
            best_auc = auc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            best_val_probabilities = val_probabilities
            if args.save_probabilities:
                best_val_daily = {}
                predict(model, val_loader, device, best_val_daily)
        elif args.daily_loss and epoch - best_epoch >= args.patience:
            break

    model.load_state_dict(best_weights)
    thresholds, _ = choose_thresholds(best_val_probabilities, validation)
    test_loader, _ = make_loader(graph, data, user_ids, [holdout], cold_id, args.batch_size, False,
                                 hops=args.hops)
    test_daily = {}
    test_probabilities = predict(model, test_loader, device, test_daily)
    if args.save_probabilities:
        np.savez_compressed(args.metrics.with_suffix(".probs.npz"), val=best_val_probabilities, test=test_probabilities,
                            val_daily=best_val_daily.get("x", 0), test_daily=test_daily.get("x", 0),
                            val_y=(validation.target_counts > 0), test_y=(holdout.target_counts > 0),
                            val_mask=validation.label_mask, test_mask=holdout.label_mask,
                            warm=holdout.user_nodes.numpy() >= 0, user_ids=user_ids)
    predictions = test_probabilities >= thresholds
    warm_users = holdout.user_nodes.numpy() >= 0
    results = {
        "relbench_commit": revision,
        "versions": {name: version(name) for name in ("relbench", "pytorch-frame", "torch",
                                                     "torch-geometric", "pyg-lib")},
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "channels": CHANNELS,
        "hops": args.hops,
        "neighbors": neighbors_by_hop(args.hops),
        "learning_rate": learning_rate,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "validation_macro_roc_auc": best_auc,
        "user_count": len(user_ids),
        "holdout_warm_count": int(warm_users.sum()),
        "graph_nodes": graph.num_nodes_dict,
        "graph_directed_edges": graph.num_edges,
        "text_columns": text_columns,
        "train_cutoffs" if args.daily_loss else "train_weeks": [taipei_date(cutoff) for cutoff in cutoffs],
        "validation_week": taipei_date(validation.cutoff),
        "holdout_week": taipei_date(holdout.cutoff),
        "thresholds": dict(zip(EVENT_COLS, thresholds.tolist())),
        "validation": evaluate_week(best_val_probabilities >= thresholds, validation),
        "holdout": evaluate_week(predictions, holdout),
        "holdout_warm_users": evaluate_week(predictions, holdout, warm_users),
        "holdout_cold_users": evaluate_week(predictions, holdout, ~warm_users),
    }
    if args.daily_loss:
        results.update(forecast=forecast, gru_hidden_size=DAILY_HIDDEN_SIZE,
                       first_optimizer_step=gradient_audit)
    if pretraining is not None:
        results.update(pretraining=pretraining, encoder_frozen=history.encoder_frozen)
        if args.encoder_finetune_lr is not None:
            results["encoder_learning_rate"] = args.encoder_finetune_lr
        if args.encoder_checkpoint:
            results.update(encoder_checkpoint=str(args.encoder_checkpoint), initial_state_sha256=initial_state_hash)
    checkpoint = args.metrics.with_suffix(".pt")
    write_atomically(checkpoint, lambda path: torch.save({"model": best_weights,
                     "column_stats": column_stats, "results": results}, path))
    write_atomically(args.metrics, lambda path: path.write_text(json.dumps(results, indent=2) + "\n"))
    print(json.dumps(results["holdout"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "data/processed/db")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/download_manifest.json")
    parser.add_argument("--missing-activity-csv", type=Path)
    parser.add_argument("--cohort", type=Path, default=ROOT / "scripts/cohort.txt")
    parser.add_argument("--upstream", type=Path, default=RUN_DIR / "upstream")
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--validation-cutoff", default="2026-11-01")
    parser.add_argument("--holdout-cutoff", default="2026-11-08")
    parser.add_argument("--min-history-hours", type=int, default=24)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--daily-loss", action="store_true",
                        help="Train the RelBench encoder and seven-step GRU with daily labels.")
    parser.add_argument("--pretrain-epochs", type=int, default=0,
                        help="Pretraining epochs on 24-hour targets. Default: 60 for --pretrain-only.")
    parser.add_argument("--encoder-checkpoint", type=Path,
                        help="Reuse a saved encoder checkpoint for the daily GRU.")
    parser.add_argument("--encoder-finetune-lr", type=float,
                        help="Update the pretrained graph encoder and projection at this learning rate.")
    parser.add_argument("--pretrain-only", action="store_true",
                        help="Stop encoder pretraining when 24-hour validation BCE plateaus.")
    mask_modes = parser.add_mutually_exclusive_group()
    mask_modes.add_argument("--masked-type-pretrain", action="store_true",
                            help="Hide historical activity rows and predict their types before daily GRU training.")
    mask_modes.add_argument("--review-feature-pretrain", action="store_true",
                            help="Mask review fields with graph edges retained, then train the daily GRU.")
    parser.add_argument("--pretrain-min-delta", type=float, default=0.0001,
                        help="Minimum absolute validation BCE decrease that resets patience in pretrain-only mode.")
    parser.add_argument("--epochs", type=int, help="Default: 10 for weekly, 30 for daily.")
    parser.add_argument("--patience", type=int, default=6,
                        help="Stop pretrain-only or daily GRU training after this many epochs without improvement.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hops", type=int, default=DEFAULT_HOPS,
                        help="Number of GraphSAGE layers and temporal sampling hops.")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-gap-csv", type=Path,
                        help="activity,start,end gaps. Days with too little observed time are dropped from the training loss only.")
    parser.add_argument("--train-gap-min-fraction", type=float, default=0.75)
    parser.add_argument("--temporal-strategy", choices=["uniform", "last"], default="uniform")
    parser.add_argument("--extra-edges", choices=["none", "head", "owner", "both"], default="none")
    parser.add_argument("--history", choices=["none", "stats", "seq", "both"], default="none",
                        help="Add per-user recent statistics and/or the last events as a sequence.")
    parser.add_argument("--history-v2", action="store_true",
                        help="Add daily rhythm, burstiness, observed-history length and source-coverage features.")
    parser.add_argument("--source-gap-csv", type=Path, default=ROOT / "data/analysis/push_unobserved_hours.csv")
    parser.add_argument("--no-time-encoding", action="store_true",
                        help="Ablation: drop the relative-time encoder from the graph model.")
    parser.add_argument("--history-pretrain", action="store_true",
                        help="Attach the --history module to the pretraining model too.")
    parser.add_argument("--no-recurrence", action="store_true",
                        help="Ablation: drop the GRU state update; each day uses initial state + day embedding.")
    parser.add_argument("--seq-edges", action="store_true",
                        help="Add previous-event edges (same user, same table) so event order is on the graph.")
    parser.add_argument("--focal-gamma", type=float, default=0.0,
                        help="Focal-loss exponent for the activity BCE (0 = plain BCE).")
    parser.add_argument("--edge-drop", type=float, default=0.0,
                        help="DropEdge: drop each sampled edge with this probability while training.")
    parser.add_argument("--event-mask", type=float, default=0.0,
                        help="Extra drop probability for edges touching the user (masks the user's own events).")
    parser.add_argument("--aug-gru", action="store_true",
                        help="Also augment the GRU-stage training loader (default: pretraining only).")
    parser.add_argument("--clip-old-days", type=int, default=0)
    parser.add_argument("--repo-group", choices=["none", "owner", "org", "both"], default="none")
    parser.add_argument("--backbone", choices=["sage", "relgnn"], default="sage",
                        help="Graph encoder: the RelBench GraphSAGE example or RelGNN.")
    parser.add_argument("--relgnn-heads", type=int, default=4)
    parser.add_argument("--rank-weight", type=float, default=0.0,
                        help="Weight of the hard-negative pairwise ranking loss added to the daily BCE.")
    parser.add_argument("--save-probabilities", action="store_true",
                        help="Write validation and holdout probabilities (weekly and daily) next to --metrics.")
    args = parser.parse_args()
    mask_pretraining = args.masked_type_pretrain or args.review_feature_pretrain
    if args.encoder_checkpoint and (not args.daily_loss or args.pretrain_only or
                                    args.pretrain_epochs or mask_pretraining):
        parser.error("Use --encoder-checkpoint with --daily-loss and without additional pretraining.")
    if args.encoder_finetune_lr is not None:
        if not np.isfinite(args.encoder_finetune_lr) or args.encoder_finetune_lr <= 0:
            parser.error("The encoder fine-tuning learning rate must be finite and positive.")
        if not args.daily_loss or args.pretrain_only or not (args.encoder_checkpoint or args.pretrain_epochs or
                                                             mask_pretraining):
            parser.error("Fine-tuning the encoder requires daily GRU training with pretrained weights.")
    if mask_pretraining and (not args.daily_loss or args.pretrain_only):
        parser.error("Masked pretraining requires --daily-loss.")
    if args.pretrain_epochs < 0 or (args.pretrain_epochs and not (args.daily_loss or args.pretrain_only)):
        parser.error("Use --daily-loss or --pretrain-only with a positive pretraining epoch count.")
    if not np.isfinite(args.pretrain_min_delta) or args.pretrain_min_delta < 0:
        parser.error("The minimum pretraining validation loss decrease must be finite and nonnegative.")
    if args.pretrain_only and args.pretrain_epochs == 0:
        args.pretrain_epochs = 60
    if mask_pretraining and args.pretrain_epochs == 0:
        args.pretrain_epochs = 30
    if args.metrics is None:
        if args.pretrain_only:
            stem = "pretrain_convergence"
        elif args.encoder_checkpoint:
            mode = "finetuned" if args.encoder_finetune_lr is not None else "frozen"
            stem = f"{args.encoder_checkpoint.stem.removesuffix('.encoder')}_{mode}"
        elif args.masked_type_pretrain:
            stem = "masked_type_gru_metrics"
        elif args.review_feature_pretrain:
            stem = "review_feature_gru_metrics"
        elif args.pretrain_epochs:
            stem = "pretrained_gru_metrics"
        else:
            stem = "daily_gru_metrics" if args.daily_loss else "metrics"
        if args.encoder_finetune_lr is not None and not args.encoder_checkpoint:
            stem += "_finetuned"
        suffix = "" if args.hops == DEFAULT_HOPS or args.encoder_checkpoint else f"_hop{args.hops}"
        args.metrics = RUN_DIR / f"{stem}{suffix}.json"
    if args.epochs is None:
        args.epochs = 30 if args.daily_loss else 10
    if min(args.epochs, args.batch_size, args.min_history_hours, args.patience, args.hops) < 1:
        parser.error("Epochs, batch size, minimum history, patience, and hops must be positive.")
    run(args)


if __name__ == "__main__":
    main()

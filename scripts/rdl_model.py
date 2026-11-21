"""RelBench-style graph encoder inputs and the user-history module used by the final model.

Builds the heterogeneous graph from the 8 tables (rows are nodes, foreign keys are edges) and
trains/evaluates the frozen-encoder + GRU model with the helpers below.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from relbench.base import Database, Table, TaskType
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal, remove_pkey_fkey
from torch import nn
from torch.nn import functional as F
from torch_frame import stype
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset

from common import ROOT
from evaluate import EVENT_COLS
from gnn_model import initialize_activity_prior
from preprocess import TABLES

RUN_DIR = ROOT / "data/model/relbench_public"
ENTITY = "Users"
FIRST_HOP_NEIGHBORS = 128
CHANNELS = 128
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
}


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
        self.extras = None  # per-user statistics module, attached in stage 2

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
    """Read the 8 tables, keep rows first seen before `cutoff`, and add one isolated cold-start user."""
    tables = {}
    for name, schema in TABLES.items():
        primary_key, row_time = schema["pk"], schema["time"]
        foreign_keys = {key: table for key, table in schema["fks"].items() if key != "merged_by_id"}
        columns = [primary_key, row_time, *foreign_keys, *FEATURE_COLUMNS[name]]
        rows = pd.read_parquet(db_dir / f"{name}.parquet", columns=columns)
        rows = rows.loc[rows[row_time].lt(cutoff)].reset_index(drop=True)
        rows[row_time] = rows[row_time].astype("datetime64[ns]")
        if name == "PullRequests":
            # The static public graph cannot represent an author edge added later.
            delayed = rows["created_at"].gt(rows[row_time])
            rows.loc[delayed, ["author_id", "created_at"]] = None
        tables[name] = Table(rows, foreign_keys, primary_key, row_time)

    for table in tables.values():
        for key, target in table.fkey_col_to_pkey_table.items():
            table.df[key] = table.df[key].where(table.df[key].lt(len(tables[target].df)).fillna(False))

    # Cold users share an isolated seed, with no edges or user ID features.
    users = tables[ENTITY]
    cold_id = len(users.df)
    users.df.loc[cold_id] = [cold_id, start - pd.Timedelta(seconds=1), True]
    return Database(tables), cold_id


def make_graph(database, text_embedder, *, fit_before):
    """Build the PK/FK graph; feature transforms are fitted on rows before `fit_before` only."""
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

    # Use RelBench's PK/FK topology and fit the feature transforms on training rows.
    # The same fitted converters handle new rows at validation and forecast time.
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


def activity_loss(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean()


def train_epoch(model, optimizer, loader, mask, device):
    model.train()
    total_loss = 0.0
    label_count = 0
    for batch in loader:
        known = mask[batch[ENTITY].input_id]
        count = int(known.sum())
        if count == 0:
            continue
        batch = batch.to(device)
        known = known.to(device)
        optimizer.zero_grad()
        loss = activity_loss(model(batch)[known], batch[ENTITY].y[known])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * count
        label_count += count
    if not label_count:
        raise ValueError("The training data has no usable activity labels.")
    return total_loss / label_count


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probabilities = np.full((len(loader.input_data.node), len(EVENT_COLS)), np.nan, dtype=np.float32)
    for batch in loader:
        indices = batch[ENTITY].input_id.numpy()
        scores = model(batch.to(device)).sigmoid()
        if scores.ndim == 3:
            # A daily probability crossing the threshold is a weekly positive.
            scores = scores.amax(dim=1)
        probabilities[indices] = scores.cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("The public loader must supply a finite prediction for every target row.")
    return probabilities


def initialize_loader_prior(model, loader, mask):
    known = mask.reshape(-1, len(EVENT_COLS))
    targets = loader.transform.target.bool().reshape_as(known)
    model.initialize_prior((targets & known).sum(dim=0).numpy(), known.sum(dim=0).numpy())

"""Reconstruct a review's hidden state from the reviewer's connected graph."""

import copy
import json
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from relbench.modeling.graph import AttachTargetTransform
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import NeighborLoader

from model_data import WEEK, taipei_date


USERS = "Users"
REVIEWS = "PRReviews"
REVIEW_STATES = ("approved", "commented", "changes_requested")
REVIEW_BATCH_SIZE = 64


def review_examples(data, user_ids, end, min_history_hours):
    """Use the first instant that both the review and its event are observable."""
    reviews = data.tables[REVIEWS].copy()
    reviews["cutoff"] = reviews[["first_seen_at", "submitted_at"]].max(axis=1)
    reviews["label"] = reviews["state"].map({name: i for i, name in enumerate(REVIEW_STATES)})
    reviewer_codes = reviews["reviewer_id"].map(data.user_codes)
    start = data.coverage.start + pd.Timedelta(hours=min_history_hours)
    eligible = (reviews["label"].notna() & reviewer_codes.isin(user_ids) &
                reviews["cutoff"].ge(start) & reviews["cutoff"].lt(end))
    return reviews.loc[eligible].sort_values(["cutoff", "review_id"])


def review_loader(graph, rows, neighbors, batch_size, *, shuffle):
    nodes = torch.tensor(rows["reviewer_id"].to_numpy(dtype=np.int64))
    times = torch.tensor(rows["cutoff"].to_numpy(dtype="datetime64[s]").astype(np.int64))
    # The sampler is inclusive here: keep the target review node at its first visible instant.
    targets = torch.tensor(rows[["review_id", "label"]].to_numpy(dtype=np.int64))
    return NeighborLoader(graph, num_neighbors=neighbors, input_nodes=(USERS, nodes),
                          input_time=times, time_attr="time", temporal_strategy="uniform",
                          transform=AttachTargetTransform(USERS, targets), batch_size=batch_size,
                          shuffle=shuffle, num_workers=0)


def find_masked_reviews(batch):
    users, reviews = batch[USERS], batch[REVIEWS]
    target_ids = users.y[:, 0]
    hidden = reviews.n_id == target_ids[reviews.batch]
    reachable = torch.zeros_like(target_ids, dtype=torch.bool)
    reachable[reviews.batch[hidden]] = True
    return hidden, reachable


class ReviewReconstruction(nn.Module):
    """Replace the target's own fields, then decode its state at the user seed."""

    def __init__(self, history):
        super().__init__()
        self.history = history
        self.mask_embedding = nn.Parameter(history.projection.weight.new_zeros(history.projection.in_features))
        self.head = nn.Linear(history.projection.out_features, len(REVIEW_STATES))

    def forward(self, batch, hidden, *, ablation=None):
        encoder = self.history.graph_encoder
        frames = batch.tf_dict
        # The masked review's original fields never reach the field encoder.
        frames[REVIEWS] = frames[REVIEWS][~hidden]
        features = encoder.encoder(frames)
        review_features = self.mask_embedding.expand(len(hidden), -1).clone()
        review_features[~hidden] = features[REVIEWS]
        features[REVIEWS] = review_features

        root_count = batch[USERS].batch_size
        if ablation == "fields":
            # Keep the topology, target mask token, and time signals.
            for name, values in features.items():
                if name == USERS:
                    features[name] = torch.cat([values[:root_count], torch.zeros_like(values[root_count:])])
                elif name == REVIEWS:
                    features[name] = values.masked_fill((~hidden)[:, None], 0)
                else:
                    features[name] = torch.zeros_like(values)

        for name, temporal in encoder.temporal_encoder(batch[USERS].seed_time, batch.time_dict,
                                                        batch.batch_dict).items():
            features[name] = features[name] + temporal
        edges = batch.edge_index_dict
        if ablation == "edges":
            edges = {relation: index[:, :0] for relation, index in edges.items()}
        user_vectors = encoder.gnn(features, edges)[USERS][:root_count]
        state = self.history.shared_user + self.history.dropout(self.history.projection(user_vectors))
        return self.head(state)


def fit_review_epoch(model, loader, weights, optimizer, device, audit_groups=None):
    model.train()
    total_loss = total_weight = 0.0
    gradients = {}
    for batch in loader:
        batch = batch.to(device)
        hidden, reachable = find_masked_reviews(batch)
        labels = batch[USERS].y[reachable, 1]
        if not labels.numel():
            continue
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch, hidden)[reachable]
        weight = weights[labels].sum()
        loss = F.cross_entropy(logits, labels, weight=weights, reduction="sum")
        (loss / weight).backward()
        if audit_groups and not gradients:
            gradients = {name: float(sum(p.grad.detach().square().sum().item() for p in parameters
                                         if p.grad is not None) ** 0.5)
                         for name, parameters in audit_groups.items()}
        optimizer.step()
        total_loss += loss.item()
        total_weight += weight.item()
    if not total_weight:
        raise ValueError("No target review was reachable from the pretraining samples.")
    return total_loss / total_weight, gradients


@torch.no_grad()
def evaluate_reviews(model, loader, device, seed, *, ablation=None):
    model.eval()
    n_states = len(REVIEW_STATES)
    loss_by_state = torch.zeros(n_states, dtype=torch.float64)
    confusion = torch.zeros((n_states, n_states), dtype=torch.long)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for batch in loader:
            batch = batch.to(device)
            hidden, reachable = find_masked_reviews(batch)
            labels = batch[USERS].y[reachable, 1]
            if not labels.numel():
                continue
            logits = model(batch, hidden, ablation=ablation)[reachable]
            loss = F.cross_entropy(logits, labels, reduction="none").cpu().double()
            y, p = labels.cpu(), logits.argmax(dim=1).cpu()
            loss_by_state.scatter_add_(0, y, loss)
            confusion += torch.bincount(y * n_states + p, minlength=n_states**2).reshape(n_states, n_states)
    totals = confusion.sum(dim=1)
    known = totals > 0
    if not known.any():
        raise ValueError("No target review was reachable from the validation samples.")
    return {"loss": (loss_by_state[known] / totals[known]).mean().item(),
            "macro_recall": (confusion.diag()[known].double() / totals[known]).mean().item(),
            "accuracy": confusion.diag().sum().item() / totals.sum().item(),
            "confusion": confusion.tolist(), "samples": totals.sum().item()}


def pretrain_review_features(history, graph, data, user_ids, val_time, neighbors, args, device, learning_rate):
    rows = review_examples(data, user_ids, val_time + WEEK, args.min_history_hours)
    train_rows = rows.loc[rows["cutoff"].lt(val_time)]
    val_rows = rows.loc[rows["cutoff"].ge(val_time)]
    if train_rows.empty or val_rows.empty:
        raise ValueError("Review field reconstruction requires historical train and validation reviews.")
    batch_size = min(args.batch_size, REVIEW_BATCH_SIZE)
    train_loader = review_loader(graph, train_rows, neighbors, batch_size, shuffle=True)
    val_loader = review_loader(graph, val_rows, neighbors, batch_size, shuffle=False)
    counts = np.bincount(train_rows["label"].to_numpy(dtype=np.int64), minlength=len(REVIEW_STATES))
    weights = torch.tensor(1 / counts.clip(min=1), dtype=torch.float32, device=device)
    model = ReviewReconstruction(history).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    audit_groups = {"neighbor_field_mlp": tuple(history.graph_encoder.encoder.encoders["PullRequests"].parameters()),
                    "graphsage": tuple(history.graph_encoder.gnn.parameters()),
                    "user_projection": tuple(history.projection.parameters())}
    best_loss = patience_loss = float("inf")
    last_improvement = 0
    records = []
    print(f"Review field reconstruction: {len(train_rows):,} train, {len(val_rows):,} validation targets",
          flush=True)
    for epoch in range(1, args.pretrain_epochs + 1):
        started = perf_counter()
        train_loss, gradients = fit_review_epoch(model, train_loader, weights, optimizer, device,
                                                audit_groups if epoch == 1 else None)
        if gradients:
            first_gradients = gradients
        validation = evaluate_reviews(model, val_loader, device, args.seed)
        val_loss = validation["loss"]
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise ValueError(f"Non-finite review reconstruction loss at epoch {epoch}.")
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_weights = copy.deepcopy(model.state_dict())
            best_validation = validation
        if val_loss < patience_loss - args.pretrain_min_delta:
            patience_loss, last_improvement = val_loss, epoch
        record = {"epoch": epoch, "loss": train_loss, "validation_loss": val_loss,
                  "validation_macro_recall": validation["macro_recall"], "seconds": perf_counter() - started}
        records.append(record)
        print(json.dumps({"stage": "review_feature_mask", **record}), flush=True)
        if epoch - last_improvement >= args.patience:
            break

    model.load_state_dict(best_weights)
    field_ablation = evaluate_reviews(model, val_loader, device, args.seed, ablation="fields")
    edge_ablation = evaluate_reviews(model, val_loader, device, args.seed, ablation="edges")
    report = {"objective": "masked_review_state", "classes": list(REVIEW_STATES),
              "feature_fit_before_taipei": taipei_date(val_time), "batch_size": batch_size,
              "train_counts": dict(zip(REVIEW_STATES, counts.tolist())), "validation_targets": len(val_rows),
              "first_batch_gradient_l2": first_gradients, "best_epoch": best_epoch,
              "validation": best_validation, "neighbor_fields_ablated": field_ablation,
              "neighbor_edges_ablated": edge_ablation, "epochs": records}
    pretext_weights = {"mask_embedding": model.mask_embedding.detach().cpu(), "head": model.head.state_dict()}
    return report, pretext_weights

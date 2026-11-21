"""Contrast complete, nonempty activity profiles on RelBench user embeddings."""

import torch
from torch.nn import functional as F


ENTITY = "Users"
TEMPERATURE = 0.2


def active_profile_loss(states, targets, user_nodes):
    """Pull together exact label matches while retaining partial label overlap."""
    if len(states) < 3:
        return states.sum() * 0, 0

    different_users = user_nodes[:, None] != user_nodes[None, :]
    overlap = targets @ targets.T
    sizes = targets.sum(dim=1)
    union = sizes[:, None] + sizes[None, :] - overlap
    same_profile = overlap == union
    positive = same_profile & different_users
    negative = ~same_profile & different_users
    valid = positive.any(dim=1) & negative.any(dim=1)
    count = int(valid.sum())
    if not count:
        return states.sum() * 0, 0

    embedding = F.normalize(states, dim=1)
    similarities = (embedding[valid] @ embedding.T) / TEMPERATURE
    positive = positive[valid]
    weights = torch.where(same_profile[valid], 1.0, 1.0 - overlap[valid] / union[valid])
    weighted = similarities + weights.clamp_min(1e-8).log()
    weighted = weighted.masked_fill(~different_users[valid], -torch.inf)
    positive_mean = (similarities * positive).sum(dim=1) / positive.sum(dim=1)
    # The softmax emphasizes different profiles that the encoder currently places close together.
    return (torch.logsumexp(weighted, dim=1) - positive_mean).mean(), count


def train_contrastive_epoch(history, optimizer, loader, mask, device, contrastive_weight):
    history.train()
    bce_total = contrastive_total = 0.0
    label_count = anchor_count = eligible_count = 0
    for batch in loader:
        known = mask[batch[ENTITY].input_id]
        count = int(known.sum())
        if not count:
            continue
        batch, known = batch.to(device), known.to(device)
        labels = batch[ENTITY].y
        states = history.initial_state(batch)
        logits = history.classifier(states)
        bce = F.binary_cross_entropy_with_logits(logits[known], labels[known])
        users = batch[ENTITY].n_id[:len(labels)]
        # Cold users share a state, so keep them in BCE without assigning them contrastive targets.
        active = known.all(dim=1) & labels.bool().any(dim=1) & (users != history.cold_id)
        active_states = states[active]
        contrastive, anchors = active_profile_loss(active_states, labels[active], users[active])

        optimizer.zero_grad(set_to_none=True)
        (bce + contrastive_weight * contrastive).backward()
        optimizer.step()
        bce_total += bce.item() * count
        contrastive_total += contrastive.item() * anchors
        label_count += count
        anchor_count += anchors
        eligible_count += len(active_states)
    if not label_count:
        raise ValueError("The training samples have no observed activity labels.")
    return {"loss": bce_total / label_count, "contrastive_loss": contrastive_total / max(1, anchor_count),
            "contrastive_anchors": anchor_count, "eligible_active_samples": eligible_count}

"""Two-layer heterogeneous GraphSAGE with a shared prediction for cold users."""

import torch
from torch import nn

from model_data import EDGE_TYPES, EVENT_COLS, WEEK


def initialize_activity_prior(classifier, counts, known):
    # Add one positive and one negative so every initial logit is finite.
    probability = (counts + 1) / (known + 2)
    bias = torch.logit(torch.as_tensor(probability, dtype=classifier.bias.dtype))
    with torch.no_grad():
        classifier.bias.copy_(bias)


class HeterogeneousSAGE(nn.Module):
    def __init__(self, node_types, hidden_size):
        super().__init__()
        self.self_transform = nn.ModuleDict({
            table: nn.Linear(hidden_size, hidden_size) for table in node_types
        })
        self.relation_transform = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False) for _ in EDGE_TYPES
        ])

    def forward(self, features, edges):
        updated = {table: transform(features[table])
                   for table, transform in self.self_transform.items()}
        for relation, transform in zip(EDGE_TYPES, self.relation_transform):
            source, _, target = relation
            source_nodes, target_nodes, degree = edges[relation]
            if source_nodes.numel() == 0:
                continue

            # Average neighbors within each relation, then add the relation's contribution.
            messages = features[source].index_select(0, source_nodes)
            neighbor_sum = torch.zeros_like(features[target])
            neighbor_sum.index_add_(0, target_nodes, messages)
            updated[target] = updated[target] + transform(neighbor_sum / degree[:, None])
        return {table: torch.relu(values) for table, values in updated.items()}


class WeeklyActivityModel(nn.Module):
    def __init__(self, feature_sizes, hidden_size, num_activities):
        super().__init__()
        self.encoder = nn.ModuleDict({
            table: nn.Linear(size, hidden_size) for table, size in feature_sizes.items()
        })
        self.layers = nn.ModuleList([
            HeterogeneousSAGE(feature_sizes, hidden_size),
            HeterogeneousSAGE(feature_sizes, hidden_size),
        ])
        self.shared_user = nn.Parameter(torch.zeros(hidden_size))
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(hidden_size, num_activities)

    def initialize_prior(self, counts, known):
        initialize_activity_prior(self.classifier, counts, known)

    @staticmethod
    def requested_users(user_vectors, snapshot):
        # Missing or disconnected users map to -1, which selects the last, zero row.
        lookup = torch.cat([user_vectors, user_vectors.new_zeros(1, user_vectors.shape[1])])
        return lookup[snapshot.user_nodes]

    def user_correction(self, snapshot):
        features = {table: torch.relu(transform(snapshot.features[table]))
                    for table, transform in self.encoder.items()}
        for layer in self.layers:
            features = layer(features, snapshot.edges)
        return self.requested_users(features["Users"], snapshot)

    def forward(self, snapshot):
        return self.classifier(self.shared_user + self.dropout(self.user_correction(snapshot)))


class DailyActivityModel(WeeklyActivityModel):
    """Use one graph encoder for all seven days and combine day vectors in order."""

    def __init__(self, feature_sizes, hidden_size, num_activities):
        super().__init__(feature_sizes, hidden_size, num_activities)
        day_size = hidden_size + len(EVENT_COLS) + 1
        self.combine_days = nn.Linear(WEEK.days * day_size, hidden_size)

    def user_correction(self, snapshot):
        days = snapshot.daily_graphs
        encode_day = super().user_correction
        embeddings = torch.stack([encode_day(day) for day in days], dim=1)
        observed = torch.stack([day.user_nodes >= 0 for day in days], dim=1)
        coverage = snapshot.daily_coverage.unsqueeze(0).expand(len(snapshot.user_nodes), -1, -1)
        sequence = torch.cat([embeddings, coverage, observed.unsqueeze(-1)], dim=-1)
        daily = torch.relu(self.combine_days(sequence.flatten(start_dim=1)))
        # Keep the existing aggregate user features as context, even on quiet weeks.
        history = torch.relu(self.encoder["Users"](snapshot.features["Users"]))
        context = self.requested_users(history, snapshot)
        daily = daily * observed.any(dim=1, keepdim=True)
        return context + daily


class NextDayActivityModel(nn.Module):
    """Unroll one next-day decoder for seven days from a single history snapshot."""

    def __init__(self, history_model, hidden_size, num_activities):
        super().__init__()
        self.history_model = history_model
        self.forecast_day = nn.Parameter(torch.zeros(WEEK.days, hidden_size))
        self.update_state = nn.GRUCell(hidden_size + num_activities, hidden_size)
        self.extras = None  # optional per-user history module (stats and/or event sequence)
        self.recurrent = True  # False: skip the GRU update, every day uses the initial state + its day embedding

    def initialize_prior(self, counts, known):
        self.history_model.initialize_prior(counts, known)

    def forward(self, snapshot):
        correction = self.history_model.user_correction(snapshot)
        if self.extras is not None:
            store = snapshot["Users"]
            correction = correction + self.extras(store, correction.size(0))
        return self.forecast_from_correction(correction)

    def forecast_from_correction(self, correction):
        """Decode a history correction already computed by the graph encoder."""
        history = self.history_model
        state = history.shared_user + history.dropout(correction)
        predictions = []
        for day in range(WEEK.days):
            position = self.forecast_day[day].expand_as(state)
            logits = history.classifier(state + position)
            predictions.append(logits)
            if self.recurrent and day + 1 < WEEK.days:
                # Use the model's previous forecast in both training and inference.
                update = torch.cat([torch.sigmoid(logits), position], dim=1)
                state = self.update_state(update, state)
        return torch.stack(predictions, dim=1)

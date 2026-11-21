"""Next-day activity decoder: a 7-step GRU unrolled from one user vector."""

import torch
from torch import nn

from model_data import EVENT_COLS, WEEK


def initialize_activity_prior(classifier, counts, known):
    # Add one positive and one negative so every initial logit is finite.
    probability = (counts + 1) / (known + 2)
    bias = torch.logit(torch.as_tensor(probability, dtype=classifier.bias.dtype))
    with torch.no_grad():
        classifier.bias.copy_(bias)


class NextDayActivityModel(nn.Module):
    """Unroll one next-day decoder for seven days from a single history snapshot."""

    def __init__(self, history_model, hidden_size, num_activities):
        super().__init__()
        self.history_model = history_model
        self.forecast_day = nn.Parameter(torch.zeros(WEEK.days, hidden_size))
        self.update_state = nn.GRUCell(hidden_size + num_activities, hidden_size)
        self.extras = None  # per-user recent-statistics module

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
            if day + 1 < WEEK.days:
                # Use the model's previous forecast in both training and inference.
                update = torch.cat([torch.sigmoid(logits), position], dim=1)
                state = self.update_state(update, state)
        return torch.stack(predictions, dim=1)

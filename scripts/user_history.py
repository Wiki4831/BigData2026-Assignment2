"""Recent-activity statistics per (user, cutoff) and the module that turns them into a GRU-state correction.

Statistics per activity: log counts in 1/3/7/14/28-day windows, log days since the last event and the share
of the last 28 days with an event; plus the share of active days and the total event count over all
activities. Only events with event_time < cutoff and row_time < cutoff are used.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from model_data import EVENT_COLS

WINDOWS = (1, 3, 7, 14, 28)
N_ACT = len(EVENT_COLS)
STATS_DIM = N_ACT * (len(WINDOWS) + 2) + 2


class HistoryInputs:
    def __init__(self, events, user_ids):
        users = pd.Index(user_ids)
        ev = events.dropna(subset=["github_user_id", "event_time", "row_time"]).copy()
        ev["u"] = users.get_indexer(ev["github_user_id"].astype("int64"))
        ev = ev[ev["u"] >= 0].sort_values(["u", "event_time"]).reset_index(drop=True)
        self.n = len(users)
        self.u = ev["u"].to_numpy()
        self.act = ev["activity"].to_numpy()
        self.t = ev["event_time"].to_numpy("datetime64[s]").astype(np.int64)
        self.r = ev["row_time"].to_numpy("datetime64[s]").astype(np.int64)

    def stats(self, cutoff):
        c = np.int64(cutoff.to_datetime64().astype("datetime64[s]").astype(np.int64))
        vis = (self.t < c) & (self.r < c)
        u, a, age = self.u[vis], self.act[vis], (c - self.t[vis]) / 86400.0
        cols = []
        for k in range(N_ACT):
            m = a == k
            uk, ak = u[m], age[m]
            for w in WINDOWS:
                cols.append(np.log1p(np.bincount(uk[ak <= w], minlength=self.n)))
            last = np.full(self.n, np.nan)
            np.fmin.at(last, uk, ak)
            cols.append(np.log1p(np.nan_to_num(last, nan=60.0)))
            day = np.floor(ak).astype(int)
            keep = day < 28
            pairs = np.unique(uk[keep].astype(np.int64) * 28 + day[keep])
            cols.append(np.bincount(pairs // 28, minlength=self.n) / 28.0)
        day = np.floor(age).astype(int)
        keep = day < 28
        pairs = np.unique(u[keep].astype(np.int64) * 28 + day[keep])
        cols.append(np.bincount(pairs // 28, minlength=self.n) / 28.0)
        cols.append(np.log1p(np.bincount(u, minlength=self.n)))
        return np.column_stack(cols).astype(np.float32)


class HistoryExtras(nn.Module):
    """Map the statistics vector to a correction for the GRU initial state."""

    def __init__(self, hidden, cold_id):
        super().__init__()
        self.cold_id = cold_id
        self.stats_net = nn.Sequential(nn.LayerNorm(STATS_DIM), nn.Linear(STATS_DIM, hidden), nn.ReLU(),
                                       nn.Linear(hidden, hidden))
        nn.init.zeros_(self.stats_net[-1].weight)
        nn.init.zeros_(self.stats_net[-1].bias)

    def forward(self, store, batch_size):
        out = self.stats_net(store.hist_stats.float())
        seed = store.n_id[:batch_size]
        return out.masked_fill((seed == self.cold_id)[:, None], 0)

"""Per-user history inputs that sit next to the graph embedding.

stats: recent-activity statistics per (user, cutoff): windowed counts, time since last event,
       active days. sequence: the last SEQ_LEN events with type and elapsed time.
Both use only events with event_time < cutoff and row_time < cutoff.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from model_data import EVENT_COLS

WINDOWS = (1, 3, 7, 14, 28)
SEQ_LEN = 32
SEQ_DIM = len(EVENT_COLS) + 3  # type one-hot, log age, log gap to the previous event, valid flag
N_ACT = len(EVENT_COLS)
STATS_DIM = N_ACT * (len(WINDOWS) + 2) + 2
V2_EXTRA = 4 + 1 + 3 + 2 + N_ACT * len(WINDOWS)  # hour buckets, weekend, gaps, account age, source coverage


class HistoryInputs:
    def __init__(self, events, user_ids, users_table=None, coverage=None):
        users = pd.Index(user_ids)
        self.coverage = coverage
        self.first_seen = None
        if users_table is not None:
            seen = users_table.drop_duplicates("github_user_id").set_index("github_user_id")["first_seen_at"]
            seen = seen.reindex(users)
            self.first_seen = seen.to_numpy("datetime64[s]").astype(np.int64)  # NaT -> very negative
            self.first_seen_valid = seen.notna().to_numpy()
        ev = events.dropna(subset=["github_user_id", "event_time", "row_time"]).copy()
        ev["u"] = users.get_indexer(ev["github_user_id"].astype("int64"))
        ev = ev[ev["u"] >= 0].sort_values(["u", "event_time"]).reset_index(drop=True)
        self.n = len(users)
        self.u = ev["u"].to_numpy()
        self.act = ev["activity"].to_numpy()
        self.t = ev["event_time"].to_numpy("datetime64[s]").astype(np.int64)
        self.r = ev["row_time"].to_numpy("datetime64[s]").astype(np.int64)
        self.start = np.searchsorted(self.u, np.arange(self.n + 1))

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
        if self.coverage is not None:
            cols.extend(self._v2(c, cutoff, u, age, vis))
        return np.column_stack(cols).astype(np.float32)

    def _v2(self, c, cutoff, u, age, vis):
        """Extra information: daily rhythm, burstiness, observed-history length, source coverage."""
        recent = age <= 28
        ur = u[recent]
        t = self.t[vis][recent]
        local = t + 8 * 3600
        bucket = (local // 3600 % 24) // 6
        total = np.maximum(np.bincount(ur, minlength=self.n), 1)
        cols = [np.bincount(ur[bucket == b], minlength=self.n) / total for b in range(4)]
        weekend = ((local // 86400 + 3) % 7) >= 5
        cols.append(np.bincount(ur[weekend], minlength=self.n) / total)
        same = ur[1:] == ur[:-1]
        gap_h = np.diff(t)[same] / 3600.0
        gu = ur[1:][same]
        cnt = np.maximum(np.bincount(gu, minlength=self.n), 1)
        cols.append(np.bincount(gu, weights=np.log1p(gap_h), minlength=self.n) / cnt)
        cols.append(np.bincount(gu[gap_h < 1], minlength=self.n) / cnt)
        cols.append(np.bincount(gu[gap_h < 24], minlength=self.n) / cnt)
        if self.first_seen is not None:
            seen = self.first_seen_valid & (self.first_seen < c)
            cols.append(np.where(seen, np.log1p(np.maximum(c - self.first_seen, 0) / 86400.0), 0))
            cols.append(seen.astype(np.float64))
        else:
            cols.extend([np.zeros(self.n), np.zeros(self.n)])
        for w in WINDOWS:
            frac = self.coverage.fraction(cutoff - pd.Timedelta(days=w), cutoff)
            cols.extend(np.full(self.n, f) for f in frac)
        return cols

    def sequence(self, cutoff):
        c = np.int64(cutoff.to_datetime64().astype("datetime64[s]").astype(np.int64))
        out = np.zeros((self.n, SEQ_LEN, SEQ_DIM), dtype=np.float16)
        for i in range(self.n):
            lo, hi = self.start[i], self.start[i + 1]
            if hi == lo:
                continue
            end = lo + np.searchsorted(self.t[lo:hi], c)
            idx = np.arange(max(lo, end - 2 * SEQ_LEN), end)
            idx = idx[self.r[idx] < c][-SEQ_LEN:]
            m = len(idx)
            if not m:
                continue
            t = self.t[idx]
            gap = np.diff(t, prepend=t[0]) / 86400.0
            block = out[i, SEQ_LEN - m:]
            block[np.arange(m), self.act[idx]] = 1
            block[:, N_ACT] = np.log1p((c - t) / 86400.0)
            block[:, N_ACT + 1] = np.log1p(gap)
            block[:, N_ACT + 2] = 1
        return out


class HistoryExtras(nn.Module):
    """Map the stats vector and the event sequence to a correction for the GRU initial state."""

    def __init__(self, mode, hidden, cold_id, v2=False):
        super().__init__()
        self.mode, self.cold_id = mode, cold_id
        if mode in ("stats", "both"):
            dim = STATS_DIM + (V2_EXTRA if v2 else 0)
            self.stats_net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.ReLU(),
                                           nn.Linear(hidden, hidden))
            nn.init.zeros_(self.stats_net[-1].weight); nn.init.zeros_(self.stats_net[-1].bias)
        if mode in ("seq", "both"):
            self.embed = nn.Linear(SEQ_DIM, 32)
            self.gru = nn.GRU(32, hidden, batch_first=True)
            self.seq_out = nn.Linear(hidden, hidden)
            nn.init.zeros_(self.seq_out.weight); nn.init.zeros_(self.seq_out.bias)

    def forward(self, store, batch_size):
        out = 0
        if hasattr(self, "stats_net"):
            out = out + self.stats_net(store.hist_stats.float())
        if hasattr(self, "gru"):
            _, h = self.gru(torch.relu(self.embed(store.hist_seq.float())))
            out = out + self.seq_out(h[-1])
        seed = store.n_id[:batch_size]
        return out.masked_fill((seed == self.cold_id)[:, None], 0)

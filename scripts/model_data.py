"""Build the model's inputs and weekly targets from the official Parquet files."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate import EVENT_COLS
from preprocess import TABLES


DAY = pd.Timedelta(days=1)
WEEK = pd.Timedelta(days=7)
HISTORY_HOURS = (1, 6, 24, 168, 336)
ACTIVITY_SOURCE = {
    "open_pr": ("PullRequests", "author_id", "created_at"),
    "open_issue": ("Issues", "author_id", "created_at"),
    "issue_comment": ("IssueComments", "author_id", "created_at"),
    "pr_review": ("PRReviews", "reviewer_id", "submitted_at"),
    "pr_review_comment": ("PRReviewComments", "author_id", "created_at"),
    "push": ("Pushes", "user_id", "created_at"),
}
EVENT_TIME = {table: occurred for table, _, occurred in ACTIVITY_SOURCE.values()}
RELATIONS = tuple(
    (table, column, target)
    for table, schema in TABLES.items()
    for column, target in schema["fks"].items()
    if (table, column) != ("PullRequests", "merged_by_id")
)
EDGE_TYPES = RELATIONS + tuple((dst, f"reverse_{key}", src) for src, key, dst in RELATIONS)


def utc_timestamp(value):
    """Interpret a date without a timezone as Taiwan time; store timestamps as UTC."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Missing timestamp: {value!r}")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Taipei")
    return timestamp.tz_convert("UTC").tz_localize(None)


def hour_boundary(value):
    timestamp = utc_timestamp(value)
    if timestamp != timestamp.floor("h"):
        raise ValueError(f"Expected a whole-hour boundary, got {value!r}")
    return timestamp


def taipei_date(timestamp):
    return timestamp.tz_localize("UTC").tz_convert("Asia/Taipei").strftime("%Y-%m-%d")


@dataclass
class Coverage:
    start: pd.Timestamp
    end: pd.Timestamp
    file_hours: pd.DatetimeIndex
    activity_hours: pd.DataFrame

    @classmethod
    def read(cls, manifest_path, missing_activity_path=None):
        manifest = json.loads(manifest_path.read_text())
        start = hour_boundary(manifest["start_utc_inclusive"])
        end = hour_boundary(manifest["end_utc_exclusive"])
        if start >= end:
            raise ValueError("The manifest must describe a non-empty time interval.")
        hours = pd.date_range(start, end, freq="h", inclusive="left")
        downloaded = pd.DatetimeIndex([
            hour_boundary(record["hour_utc"])
            for record in manifest["files"]
            if record["status"] in {"downloaded", "existing"}
        ])
        present = hours.isin(downloaded)
        activity_hours = pd.DataFrame({event: present for event in EVENT_COLS}, index=hours)
        if missing_activity_path is not None:
            gaps = pd.read_csv(missing_activity_path, dtype=str)
            if set(gaps.columns) != {"activity", "start", "end"}:
                raise ValueError("Coverage CSV must have columns activity,start,end.")
            for gap in gaps.itertuples(index=False):
                if gap.activity not in EVENT_COLS:
                    raise ValueError(f"Unknown activity in coverage CSV: {gap.activity}")
                gap_start, gap_end = hour_boundary(gap.start), hour_boundary(gap.end)
                if gap_start >= gap_end:
                    raise ValueError(f"Invalid coverage interval: {gap.start} to {gap.end}")
                activity_hours.loc[(hours >= gap_start) & (hours < gap_end), gap.activity] = False
        return cls(start, end, hours[present], activity_hours)

    def fraction(self, start, end):
        """Return six coverage fractions, including hours outside the manifest as missing."""
        expected_hours = (end - start) / pd.Timedelta(hours=1)
        covered = self.activity_hours.loc[(self.activity_hours.index >= start) &
                                          (self.activity_hours.index < end)].sum()
        return covered.to_numpy(dtype=np.float32) / expected_hours

    def has_history(self, cutoff, hours):
        return ((self.file_hours >= cutoff - pd.Timedelta(hours=hours)) &
                (self.file_hours < cutoff)).sum() >= hours


@dataclass
class Snapshot:
    """A graph and its requested users at one cutoff. Request order is preserved."""

    cutoff: pd.Timestamp
    features: dict
    edges: dict
    user_nodes: torch.Tensor
    last_week: np.ndarray
    target_counts: np.ndarray | None = None
    label_mask: np.ndarray | None = None
    daily_graphs: tuple["Snapshot", ...] = ()
    daily_coverage: torch.Tensor | None = None
    next_day_counts: np.ndarray | None = None
    next_day_mask: np.ndarray | None = None

    def to(self, device):
        self.features = {table: values.to(device) for table, values in self.features.items()}
        self.edges = {relation: tuple(values.to(device) for values in edge)
                      for relation, edge in self.edges.items()}
        self.user_nodes = self.user_nodes.to(device)
        if self.daily_graphs:
            for day in self.daily_graphs:
                day.to(device)
            self.daily_coverage = self.daily_coverage.to(device)
        return self


def selected_columns(table):
    schema = TABLES[table]
    columns = {schema["pk"], schema["time"]}
    columns.update(key for src, key, _ in RELATIONS if src == table)
    if table in EVENT_TIME:
        columns.add(EVENT_TIME[table])
    extra = {
        "Users": {"github_user_id", "is_cohort"},
        "Issues": {"is_pull_request"},
        "PRReviews": {"state"},
        "PRReviewComments": {"subject_type"},
        "Pushes": {"ref_type", "is_new_ref"},
    }
    return sorted(columns | extra.get(table, set()))


def count_events(events, id_column, user_ids):
    """Return a [user, activity] count matrix in the supplied user order."""
    counts = np.zeros((len(user_ids), len(EVENT_COLS)), dtype=np.int32)
    positions = user_ids.get_indexer(events[id_column])
    valid = positions >= 0
    np.add.at(counts, (positions[valid], events["activity"].to_numpy()[valid]), 1)
    return counts


def boolean_feature(series):
    return series.fillna(False).to_numpy(dtype=np.float32)


def age_hours(cutoff, timestamps):
    return (cutoff - timestamps).dt.total_seconds().to_numpy(dtype=np.float64) / 3600


class ActivityData:
    def __init__(self, db_dir: Path, coverage: Coverage):
        self.coverage = coverage
        self.tables = {}
        for table, schema in TABLES.items():
            rows = pd.read_parquet(db_dir / f"{table}.parquet", columns=selected_columns(table))
            row_time = rows[schema["time"]]
            if row_time.ge(coverage.end).any() or row_time.lt(coverage.start).any():
                raise ValueError(f"{table} has rows outside the manifest. Use matching inputs.")
            self.tables[table] = rows
        # Optional training-only gaps (see --train-gap-csv): days whose observed fraction is below
        # train_gap_min_fraction are dropped from the daily loss, positives included.
        self.train_gaps = None
        self.train_gap_min_fraction = 0.75
        self.user_codes = self.tables["Users"].set_index("user_id")["github_user_id"]
        self.events = self._read_events()

    def _read_events(self):
        records = []
        for activity_index, activity in enumerate(EVENT_COLS):
            table, actor, event_time = ACTIVITY_SOURCE[activity]
            rows = self.tables[table]
            if activity == "open_issue":
                rows = rows.loc[rows["is_pull_request"].eq(False).fillna(False)]
            event = pd.DataFrame({"row_id": rows[TABLES[table]["pk"]],
                                  "user_id": rows[actor], "event_time": rows[event_time],
                                  "row_time": rows[TABLES[table]["time"]]})
            event = event.dropna(subset=["user_id", "event_time", "row_time"])
            event = event.loc[event["event_time"].ge(self.coverage.start) &
                              event["event_time"].lt(self.coverage.end)].copy()
            event["github_user_id"] = event["user_id"].map(self.user_codes)
            event["activity"] = activity_index
            records.append(event)
        return pd.concat(records, ignore_index=True)

    def _user_features(self, cutoff, users, requested_users):
        user_ids = pd.Index(users["user_id"])
        prior = self.events.loc[self.events["row_time"].lt(cutoff) &
                                self.events["event_time"].lt(cutoff)]
        features = []
        for hours in HISTORY_HOURS:
            start = cutoff - pd.Timedelta(hours=hours)
            counts = count_events(prior.loc[prior["event_time"].ge(start)], "user_id", user_ids)
            fraction = self.coverage.fraction(start, cutoff)
            # A zero count means something different in an unobserved window.
            features.extend([np.log1p(counts), np.broadcast_to(fraction, counts.shape)])

        last = prior.groupby(["user_id", "activity"])["event_time"].max().unstack()
        last = last.reindex(index=user_ids, columns=range(len(EVENT_COLS)))
        elapsed = cutoff.to_datetime64() - last.to_numpy(dtype="datetime64[ns]")
        hours_since_last = elapsed / np.timedelta64(1, "h")
        known = np.isfinite(hours_since_last)
        features.extend([np.log1p(np.where(known, hours_since_last, 0) / 24), known])

        recent = prior.loc[prior["event_time"].ge(cutoff - WEEK)].copy()
        recent["local_day"] = (recent["event_time"] + pd.Timedelta(hours=8)).dt.floor("D")
        active_days = recent.groupby("user_id")["local_day"].nunique().reindex(user_ids, fill_value=0)
        features.append(active_days.to_numpy()[:, None] / 7)
        last_week = count_events(recent, "github_user_id", requested_users)
        return np.column_stack(features), last_week > 0

    def _node_features(self, table, rows, cutoff, visible_prs):
        row_age = age_hours(cutoff, rows[TABLES[table]["time"]])
        event_age = age_hours(cutoff, rows[EVENT_TIME.get(table, TABLES[table]["time"])])
        known = np.isfinite(event_age) & (event_age > 0)
        features = [np.log1p(row_age / 24), np.log1p(np.where(known, event_age, 0) / 24), known]

        if table == "PRReviews":
            pr_openings = visible_prs.set_index("pr_id")["created_at"]
            opened = pr_openings.reindex(rows["pr_id"]).to_numpy(dtype="datetime64[ns]")
            submitted = rows["submitted_at"].to_numpy(dtype="datetime64[ns]")
            lag = (submitted - opened) / np.timedelta64(1, "h")
            valid_lag = known & np.isfinite(lag) & (lag >= 0)
            features.extend([np.log1p(np.where(valid_lag, lag, 0) / 24), valid_lag])
            features.extend(boolean_feature(rows["state"].eq(state))
                            for state in ("approved", "commented", "changes_requested"))
        if table == "Issues":
            features.append(boolean_feature(rows["is_pull_request"]))
        if table == "PRReviewComments":
            features.append(boolean_feature(rows["subject_type"].eq("line")))
        if table == "Pushes":
            features.extend([boolean_feature(rows["ref_type"].eq("tag")),
                             boolean_feature(rows["is_new_ref"])])
        return np.column_stack(features)

    @staticmethod
    def _edge_positions(table, key, rows, target_index, cutoff):
        targets = rows[key].map(target_index)
        valid = targets.notna().to_numpy(copy=True)
        if table in {"PullRequests", "Issues"} and key == "author_id":
            # An opening recorded at or after the cutoff cannot establish an author yet.
            valid &= ~rows["created_at"].ge(cutoff).to_numpy()
        return np.flatnonzero(valid), targets.loc[valid].to_numpy(dtype=np.int64, copy=True)

    def _edges(self, visible, positions, cutoff):
        edges = {}
        has_evidence = np.zeros(len(visible["Users"]), dtype=bool)
        for src, key, dst in RELATIONS:
            source_nodes, target_nodes = self._edge_positions(src, key, visible[src], positions[dst], cutoff)
            if dst == "Users":
                has_evidence[target_nodes] = True
            edges[(src, key, dst)] = self._edge_tensors(source_nodes, target_nodes, len(visible[dst]))
            edges[(dst, f"reverse_{key}", src)] = self._edge_tensors(target_nodes, source_nodes, len(visible[src]))
        return edges, has_evidence

    @staticmethod
    def _edge_tensors(sources, destinations, node_count):
        degree = np.bincount(destinations, minlength=node_count).clip(min=1)
        return torch.from_numpy(sources), torch.from_numpy(destinations), torch.from_numpy(degree.astype(np.float32))

    def _targets(self, cutoff, user_ids, *, by_day=False, horizon=WEEK):
        end = cutoff + horizon
        if cutoff < self.coverage.start or end > self.coverage.end:
            raise ValueError(f"Label window beginning {taipei_date(cutoff)} is outside the manifest.")
        future = self.events.loc[self.events["event_time"].ge(cutoff) &
                                 self.events["event_time"].lt(end) & self.events["row_time"].lt(end)]
        if by_day:
            # Use one visibility deadline so daily positives combine into the window label.
            day_counts, day_coverage = [], []
            for day_start in pd.date_range(cutoff, end, freq="D", inclusive="left"):
                day_end = day_start + DAY
                day = future.loc[future["event_time"].ge(day_start) & future["event_time"].lt(day_end)]
                day_counts.append(count_events(day, "github_user_id", user_ids))
                day_coverage.append(self.coverage.fraction(day_start, day_end) == 1)
                if self.train_gaps is not None:
                    day_coverage[-1] = day_coverage[-1] & (self.train_gaps.fraction(day_start, day_end) >=
                                                           self.train_gap_min_fraction)
            counts = np.stack(day_counts, axis=1)
            if self.train_gaps is not None:
                return counts, np.broadcast_to(np.stack(day_coverage), counts.shape).copy()
            return counts, (counts > 0) | np.stack(day_coverage)

        counts = count_events(future, "github_user_id", user_ids)
        # A recorded positive is known even if the rest of its activity window is incomplete.
        label_mask = (counts > 0) | (self.coverage.fraction(cutoff, end) == 1)
        return counts, label_mask

    def _visible_rows(self, cutoff, history_start):
        visible = {table: rows.loc[rows[TABLES[table]["time"]].lt(cutoff)]
                   for table, rows in self.tables.items()}
        if history_start is None:
            return visible
        return self._keep_recent_rows(visible, history_start)

    @staticmethod
    def _keep_recent_rows(visible, history_start):
        # Start with recent activity rows, then follow their foreign keys to keep
        # older context, for example a PR that received a recent review.
        ids = {table: set(rows[TABLES[table]["pk"]]) for table, rows in visible.items()}
        selected = {
            table: set(rows.loc[rows[TABLES[table]["time"]].ge(history_start), TABLES[table]["pk"]])
            if table in EVENT_TIME else set()
            for table, rows in visible.items()
        }
        while True:
            size = sum(map(len, selected.values()))
            for source, column, target in RELATIONS:
                rows = visible[source]
                references = rows.loc[rows[TABLES[source]["pk"]].isin(selected[source]), column].dropna()
                selected[target].update(ids[target].intersection(references))
            if sum(map(len, selected.values())) == size:
                break
        return {table: rows.loc[rows[TABLES[table]["pk"]].isin(selected[table])]
                for table, rows in visible.items()}

    def snapshot(self, cutoff, user_ids, labeled=False, *, history_start=None, daily_targets=False,
                 preserve_user_history=False, target_horizon=WEEK):
        user_ids = pd.Index(user_ids, dtype="int64")
        full_history = None
        if preserve_user_history:
            full_history = self._visible_rows(cutoff, None)
            visible = self._keep_recent_rows(full_history, history_start)
            visible["Users"] = full_history["Users"]
        else:
            visible = self._visible_rows(cutoff, history_start)
        # Remap original table IDs to contiguous indices within this snapshot.
        positions = {table: pd.Series(np.arange(len(rows)), index=rows[TABLES[table]["pk"]])
                     for table, rows in visible.items()}
        features = {table: self._node_features(table, rows, cutoff, visible["PullRequests"])
                    for table, rows in visible.items()}
        user_features, last_week = self._user_features(cutoff, visible["Users"], user_ids)
        features["Users"] = np.column_stack([features["Users"], user_features])
        edges, has_evidence = self._edges(visible, positions, cutoff)
        if full_history is not None:
            # A previously observed user still gets their own history features,
            # even if the recent graph has no neighboring events for them.
            for table, key, target in RELATIONS:
                if target == "Users":
                    _, known_users = self._edge_positions(table, key, full_history[table], positions["Users"], cutoff)
                    has_evidence[known_users] = True
        nodes = pd.Index(visible["Users"]["github_user_id"]).get_indexer(user_ids).copy()
        present = nodes >= 0
        nodes[present] = np.where(has_evidence[nodes[present]], nodes[present], -1)
        targets, mask = self._targets(cutoff, user_ids, by_day=daily_targets,
                                      horizon=target_horizon) if labeled else (None, None)
        daily_counts, daily_mask = None, None
        if labeled and daily_targets:
            daily_counts, daily_mask = targets, mask
            targets = daily_counts.sum(axis=1)
            mask = (targets > 0) | daily_mask.all(axis=1)
        return Snapshot(cutoff, {table: torch.tensor(values, dtype=torch.float32)
                                 for table, values in features.items()},
                        edges, torch.from_numpy(nodes), last_week, targets, mask,
                        next_day_counts=daily_counts, next_day_mask=daily_mask)

    def recent_snapshot(self, cutoff, user_ids, labeled=False, *, days, daily_targets=False):
        return self.snapshot(cutoff, user_ids, labeled, history_start=cutoff - days * DAY,
                             daily_targets=daily_targets, preserve_user_history=True)

    def daily_snapshot(self, cutoff, user_ids, labeled=False, *, daily_targets=False):
        """Keep the weekly targets and add seven ordered graphs ending at the cutoff."""
        snapshot = self.snapshot(cutoff, user_ids, labeled, daily_targets=daily_targets)
        day_starts = [cutoff - days * DAY for days in range(WEEK.days, 0, -1)]
        snapshot.daily_graphs = tuple(self.snapshot(day + DAY, user_ids, history_start=day)
                                      for day in day_starts)
        snapshot.daily_coverage = torch.from_numpy(np.stack([
            self.coverage.fraction(day, day + DAY) for day in day_starts
        ]))
        return snapshot

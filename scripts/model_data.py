"""Build the activity events, coverage and visibility checks from the official Parquet files."""

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


def selected_columns(table):
    schema = TABLES[table]
    columns = {schema["pk"], schema["time"]}
    columns.update(key for src, key, _ in RELATIONS if src == table)
    if table in EVENT_TIME:
        columns.add(EVENT_TIME[table])
    extra = {"Users": {"github_user_id"}, "Issues": {"is_pull_request"}}
    return sorted(columns | extra.get(table, set()))


def count_events(events, id_column, user_ids):
    """Return a [user, activity] count matrix in the supplied user order."""
    counts = np.zeros((len(user_ids), len(EVENT_COLS)), dtype=np.int32)
    positions = user_ids.get_indexer(events[id_column])
    valid = positions >= 0
    np.add.at(counts, (positions[valid], events["activity"].to_numpy()[valid]), 1)
    return counts


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

    @staticmethod
    def _edge_positions(table, key, rows, target_index, cutoff):
        targets = rows[key].map(target_index)
        valid = targets.notna().to_numpy(copy=True)
        if table in {"PullRequests", "Issues"} and key == "author_id":
            # An opening recorded at or after the cutoff cannot establish an author yet.
            valid &= ~rows["created_at"].ge(cutoff).to_numpy()
        return np.flatnonzero(valid), targets.loc[valid].to_numpy(dtype=np.int64, copy=True)

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
            counts = np.stack(day_counts, axis=1)
            return counts, (counts > 0) | np.stack(day_coverage)

        counts = count_events(future, "github_user_id", user_ids)
        # A recorded positive is known even if the rest of its activity window is incomplete.
        label_mask = (counts > 0) | (self.coverage.fraction(cutoff, end) == 1)
        return counts, label_mask

    def _visible_rows(self, cutoff):
        return {table: rows.loc[rows[TABLES[table]["time"]].lt(cutoff)] for table, rows in self.tables.items()}

    def user_nodes(self, cutoff, user_ids):
        """Row of each requested user in the graph visible at `cutoff`, or -1 for a user without any edge."""
        user_ids = pd.Index(user_ids, dtype="int64")
        visible = self._visible_rows(cutoff)
        positions = {table: pd.Series(np.arange(len(rows)), index=rows[TABLES[table]["pk"]])
                     for table, rows in visible.items()}
        has_evidence = np.zeros(len(visible["Users"]), dtype=bool)
        for src, key, dst in RELATIONS:
            _, target_nodes = self._edge_positions(src, key, visible[src], positions[dst], cutoff)
            if dst == "Users":
                has_evidence[target_nodes] = True
        nodes = pd.Index(visible["Users"]["github_user_id"]).get_indexer(user_ids).copy()
        present = nodes >= 0
        nodes[present] = np.where(has_evidence[nodes[present]], nodes[present], -1)
        return torch.from_numpy(nodes)

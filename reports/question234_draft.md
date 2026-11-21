# Questions 2–4 and Data Source: first draft

Status: draft for discussion. Numbers marked (1 seed) come from seed 42 only; the seed noise on one setting is 1.5–2 points, so gaps under about 1 point are not evidence. "Holdout 11/8" means train on weeks before 11/1, choose epoch and thresholds on week 11/1, score week 11/8.

## 3.1 Data Source (add to README.md)

> **Data Source.** The raw events come from [GH Archive](https://www.gharchive.org/) (hourly files, 2026-10-01 to 2026-11-18 UTC). All tables are produced by the provided `scripts/preprocess.py`.

The README already has a "Data Claim" section naming GH Archive; this adds the explicit "Data Source" heading the grader asks for.

## Question 2: Relationships across tables (3%)

**Baseline.** Each of the 8 tables is a node type. Each foreign key is a typed edge, and each edge gets a reverse edge. There are 19 FK columns in the graph (the post-hoc `merged_by_id` is dropped), so 38 edge types. Every table has its own column encoder (numbers, categories, booleans, timestamps; masked text through GloVe), so a node keeps its own attributes instead of being a bare ID.

**What we did to model the relations better.**

1. **Multi-hop, time-respecting neighbour sampling.** For each user and each cutoff we sample only rows already visible before the cutoff, 3 hops with 128/64/32 neighbours. Three hops reach paths such as `user → push → repo → other user's PR → review`, which is how a user is linked to other users through shared repos and PRs. Hop count (1 seed, holdout 11/8): 2 hops 37.84%, 3 hops 38.06%, 5 hops 37.21%. Three hops was kept.
2. **Relation-specific message passing.** Each edge type has its own transform; messages are summed per relation (RelBench GraphSAGE, batch norm). The step from our earlier hand-built GraphSAGE to this RelBench-style encoder was the largest gain in the project: 33.49% → 35.98% (1 seed, holdout 11/8).
3. **Self-supervised pretraining over the whole graph.** The encoder is pretrained on next-day activity prediction for 11 daily cutoffs, then frozen while the 7-day GRU is trained. 36.22% → 37.84% (1 seed, holdout 11/8). It lets the encoder learn from every user's relations before the weekly labels are used.
4. **A shared cold-start user.** Users with no visible row or edge share one isolated seed node, so the model never sees a user ID (4,882 of 9,000 users are in this group on 11/8).

**Tried and did not help (report these honestly).**

| Idea | Result (3 seeds, mean score) |
|---|---|
| RelGNN backbone (atomic routes, composite message passing) | 35.89% vs SAGE 39.06% (holdout 11/8) |
| Owner and org group nodes (Repository → Owners/Orgs edges) | holdout 11/1 36.76 → 37.02; holdout 11/8 39.06 → 37.66 |
| Repo recent-heat features, sample thresholds, ranking loss | no stable gain in two holdouts |

`Issues.assignee_id → Users` was already an edge in the graph.

## Question 3: Temporal information (5%)

1. **Visibility by cutoff.** Every row has a first-visible time. For a cutoff `t` only rows with `first_seen_at < t` enter the graph and the sampler. Fields that change after the event (`merged_at`, `closed_at`, `dismissed_at`, `merged_by_id`) are removed, and PR authors that become known after the row is first seen are masked, so the model sees only what was known at `t`. This was checked on 54,000 user samples for order and sampled node times.
2. **Relative time encoding.** RelBench's temporal encoder turns `cutoff − row time` into sine/cosine position features per node type and adds it to the node embedding, so the model knows how long ago each neighbour appeared. Our own added time branch on the older model gave no gain (37.94 vs 37.06 validation, holdout 34.61 vs 34.60); the sampler-based relative time in the RelBench model is what we use.
3. **Temporal neighbour sampling** (`time_attr`, uniform within the visible window), so an old and a new event are not mixed with future rows.
4. **Daily structure.** Training uses daily cutoffs (23 to 28 per run) instead of one cutoff per week, with a next-day BCE loss. The history embedding initialises a 7-step GRU, one step per forecast day; each step receives the previous day's predicted probabilities and a learned day-position vector. The weekly probability is the maximum over the 7 daily probabilities. 35.98% → 36.22% (1 seed) from the weekly head to the daily GRU, and the larger gain came with pretraining.
5. **Taiwan-time week alignment.** All cutoffs sit on Taiwan-time Sundays, the same as the test week (2026-11-22 to 11-29).
6. **Source coverage over time.** A large part of the push signal is a data-collection effect, not user behaviour (592 of 1,152 hours have almost no push; the weekly push-user share falls from 24% to 3%). We tested masking the unobserved hours from the training loss: push AP rose about 0.02 in both holdouts but the total score was +0.3 and −0.3, so it is reported as a diagnostic and not adopted. The README says an activity missing from GH Archive during the test week is not scored, so the 5-activity score is reported next to the 6-activity score.

## Question 4: Framework figure and full pipeline (5%)

Proposed figure (to draw; layout in order, left to right):

```text
GH Archive hourly files
   │  scripts/preprocess.py (provided)
   ▼
8 tables (Users, Repositories, PullRequests, Issues, IssueComments,
          PRReviews, PRReviewComments, Pushes)  ── PK/FK graph, 19 FK + reverse = 38 edge types
   │  for each cutoff t (Taiwan Sunday 00:00, daily for training):
   │  keep rows with first_seen_at < t; temporal sampler, 3 hops (128/64/32)
   ▼
Column encoders (PyTorch Frame, GloVe for text) + relative-time encoding
   ▼
Heterogeneous GraphSAGE (3 layers, sum per relation, batch norm)   ← pretrained on next-day BCE, then frozen
   ▼  user embedding (128) → projection (64) + shared user vector
7-step GRU (one step per day, input: previous day's probabilities + day position)
   ▼
7 daily probabilities × 6 activities → weekly probability = max over days
   ▼
per-activity thresholds chosen on the validation week → 0/1 for each of the 6 activities
```

Pipeline steps for the text:

1. Download the hourly files and run the provided preprocessing (8 tables, 9,000-user cohort).
2. Build visibility-aware graphs at daily cutoffs; labels for a cutoff are the activities in `[t, t+1 day)`, and for the weekly validation `[t, t+7 days)`.
3. Pretrain the graph encoder on next-day prediction, freeze it, train the GRU head with daily BCE.
4. On the validation week choose, per activity, the probability threshold that maximises F1, and the checkpoint by the validation score.
5. For the final run, repeat with the validation week moved to the last full week before the test and predict the test week for the 4,465 users in `submission/test_input.csv`; users not in the graph use the shared cold-start prediction.
6. `scripts/evaluate.py` checks the output format. The TAs rerun the same preprocessing and code to reproduce `output_{studentID}.csv`.

## Open points to decide

- The final model is still the SAGE variant (3-hop, pretrain, GRU). Averaging three seeds' probabilities gave 37.88% / 39.18% on the two holdouts, +0.1 to +1.1 over single runs; we could ship the seed average, which also makes reproduction easier to explain.
- The final training run and the official validation week (11/15) have not been produced; the data ends 2026-11-18 08:00 Taiwan time.
- Figure still needs drawing (draw.io, like Question 1).
- Reproducibility: the pipeline needs `scripts/relbench_asg2.py` to accept `--predict` and write `output_{studentID}.csv`; this should be checked before the deadline.

## Evidence table to cite (added 2026-11-20)

Source: `reports/rdl/variants_report.md`. 5-class score, mean of 3 seeds; seed noise 0.5–1 point.

- Q2 (relations): 19 FKs (+reverse = 38 edge types) with 3-hop sampling is the base. Adding PR→repo and repo→owner edges: -1.0 (2026), -0.3 (2025-12). Owner/org group nodes: -0.6. RelGNN: about -3. So more relations did not help; say so plainly.
- Q3 (time): relative time encoding on: +0.9 (2026), +0.5 (2025-12). Uniform temporal sampling beats latest-neighbor: +0.6/+0.4. Per-day cutoffs with next-day pretraining: no-pretraining -1.2; weekly label head -1.0. User recent-activity stats (windows 1–28d): +0.2/+1.0.
- Q4: figure should show frozen pretrained encoder → user vector → stats module → GRU over 7 days → max over days → per-activity thresholds.

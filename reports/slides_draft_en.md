# Slide content draft (English, 16 pages max, to be moved into the official template)

Rules: slide text is English; every number below comes from `reports/rdl/variants_report.md`
(3 seeds, mean; "5-class" = score excluding push; seed noise is 0.5–1 point, so differences below 1 point
are not evidence). Items marked **[TBD]** must be filled after the final run; do not present them before.
Time plan (18 min): slides 1–4 = 2.5 min, 5–6 = 4, 7–9 = 3.5, 10–13 = 5, 14–16 = 3.

---
## 1. Title
- Predicting GitHub user activity with relational deep learning
- Name, student ID 314551130, course, date
- Code: public GitHub repository **[TBD: URL]**

## 2. Task and data source
- Predict, for 4,465 test users, which of 6 activities they do in 2026-11-22 00:00 – 11-29 00:00 (Taiwan time): open_pr, open_issue, issue_comment, pr_review, pr_review_comment, push
- Score = (macro-F1 + micro-F1) / 2
- **Data Source: GH Archive (https://www.gharchive.org/)**, hourly files, tables built by the provided `preprocess.py` (8 tables, 9,000-user cohort)
- Our approach: treat the database as a graph (relational deep learning, RDL), no hand-made table joins

## 3. Q1 – Schema graph and data period
- Figure: schema graph, 8 tables, 8 primary keys, 20 foreign keys (`reports/question1/schema-refined.preview.png`)
- Self-reference: PRReviewComments.in_reply_to_id → comment_id; parallel edges to Users (PR author/merger, Issue author/assignee)
- Data period used: 2026-10-01 00:00 UTC to the last completed UTC day before the test cutoff (2026-11-21 16:00 UTC) **[TBD: confirm final end hour]**
- We use 19 of the 20 foreign keys: `merged_by_id` is only known after the event, so it is dropped

## 4. Why the task is hard
- Sparse labels: most users do few activities in a week
- Cold start: on 11/8, 4,882 of 9,000 users have no visible row before the cutoff
- Data-collection effect: 592 of 1,152 local hours have almost no push (GH Archive source differences), so push is partly a collection artifact; the README says an activity missing from GH Archive in the test week is not scored → we report the 6-activity score and a 5-activity diagnostic (without push)
- No ID features: the model cannot memorize users

## 5. Q4 – Framework figure
- Figure: `reports/figures/framework_draft.pptx` (3 rows: data → graph; time-respecting subgraph → encoder; forecast → 0/1)
- Colors: grey = data/fixed, blue = pretrained then frozen, orange = trained in stage 2
- One sentence: for every (user, cutoff) we sample a subgraph of rows seen before the cutoff, encode it into a user vector, then unroll 7 days

## 6. Q4 – Full pipeline and reproducibility
1. Download hourly GH Archive files; run the provided `preprocess.py`
2. Build one graph: each row is a node with a timestamp, each foreign key is an edge (19 FKs + reverse = 38 edge types)
3. Per (user, cutoff): temporal neighbor sampling, 3 hops (128/64/32 neighbors), only rows before the cutoff
4. RelBench column encoders + relative-time encoding → heterogeneous GraphSAGE (3 layers) → user vector
5. Pretrain the encoder on next-day activity prediction, then freeze it
6. Stage 2: initial state = shared vector + user vector (+ recent-statistics module) → 7-step GRU, one step per forecast day; daily BCE loss
7. Weekly probability = max over the 7 days; per-activity thresholds chosen on a validation week; one seed for the final run
8. `bash scripts/run_final.sh STUDENT_ID` runs download → preprocess → train → predict → format check

## 7. Q2 – Relations across tables (design)
- RDL vs CNN/RNN/GNN: input is a multi-table database; rows are nodes, foreign keys are edges, and a prediction may only use rows before its time
- Typed relations: 8 node types, 38 edge types, one transform per edge type (sum aggregation)
- Multi-hop: 3 hops link a user to other users through shared repos, PRs and issues (2 hops vs 3 hops: 0.0 difference, so 3 kept for coverage)
- Cold-start users share one isolated seed node (no user ID is ever used)
- Node features come from column content (categories, numbers, timestamps, GloVe text embeddings); repositories have no feature columns, so repository nodes differ only by structure

## 8. Q2 – Self-supervised pretraining
- Pretext task: predict the next day's activities from the sampled subgraph (BCE), on daily cutoffs
- Frozen afterwards; stage 2 only trains the GRU, shared vector and statistics module
- Without pretraining: −1.2 (5-class, 2026). Fine-tuning the encoder in stage 2: −1.4 (2026), −0.7 (2025-12 check)
- Putting the statistics module into pretraining: −1.3 → add it after pretraining

## 9. Q2 – What did not help (table)
| Idea | 2026 | 2025-12 |
|---|---|---|
| Extra edges (PR head repo, repo owner) | −1.0 | −0.3 |
| Owner / org group nodes | −0.6 | – |
| RelGNN backbone (11/8 only) | about −3 | – |
| DropEdge 0.2 / user-event masking 0.3 | −0.8 / −1.2 | −0.3 / 0.0 |
| Previous-event edges (event order in the graph) | −0.8 | −0.3 |
- Reading: the relation signal in the existing graph looks saturated; gains came from encoding and pretraining, not from more edges (our interpretation, not tested further)

## 10. Q3 – Time visibility (leakage control)
- A row is visible only if its first-seen time is before the cutoff
- Fields that become known later (`merged_at`, `merged_by_id`, `closed_at`, `state_reason`, `dismissed_at`) are removed; PR authors known only later are masked
- Example (3-table sketch): the same user has a different subgraph at each cutoff
- Training/validation are split by time; thresholds chosen on a validation week, never on the test week

## 11. Q3 – Time encoding and sampling
- Relative time = cutoff − row time (days) → sinusoidal position encoding → linear layer → added to the node vector
- Without it: −0.9 (2026), −0.5 (2025-12)
- Uniform temporal sampling vs most-recent neighbors: most-recent is −0.6 / −0.4
- The graph itself has no order between events; order is only carried by each event's relative time

## 12. Q3 – Daily cutoffs, GRU and recent statistics
- Daily cutoffs instead of one per week: 7× more training samples; next-day BCE; 7-step GRU unroll, weekly probability = max over days
- Weekly head instead (no daily loss): −1.0 (confounded with fewer cutoffs)
- Removing only the GRU state update: −0.7 (2026), −0.3 (2025-12) → slight help, weak evidence
- Recent-statistics module (counts in 1/3/7/14/28 days per activity, time since last event, active days): +0.2 (2026), +1.0 (2025-12); with 6-class score +0.7 / +0.8 → used in the final model

## 13. Q3 – Time-related negative results and the data gap
- Last 32 events through a GRU: −0.5 / +0.2; richer statistics (hour of day, weekend share, source coverage): 0.0 (6-class −1.9) → not used
- Training aligned to the data gap: −0.4 (5-class) in the backtest; the final run still trains with the same gap as the test because the real forecast has one (files end before the cutoff) **[TBD: gap in hours]**
- Focal loss: −0.4 / −0.9; flattening the output curve changes nothing at prediction because thresholds are re-tuned

## 14. Results – all variants (one table)
Columns: variant, 2026 5-class diff, 2026 6-class diff, 2025-12 5-class diff (source: `reports/rdl/variants_tables.md`)
- Helps: pretraining (+1.2), relative time encoding (+0.9 / +0.5), recent statistics (+0.2 / +1.0)
- Neutral: hop 2 vs 3, event sequence, GRU update (small)
- Hurts: fine-tuning, extra edges, group nodes, RelGNN, augmentation, previous-event edges, focal loss
- Reference on 2025-12 (test week 12/21, 5-class): GNN 48.8, copy-last-week 50.4, gradient boosting 49.3 → the GNN is on par with simple baselines there (be explicit)
- (2025-12 data was processed with the provided script and used only for these checks, not for the final model) **[decide: keep or remove]**

## 15. Final model and limitations
- Final model: pretrained frozen GraphSAGE encoder + statistics module + 7-step GRU, single seed (42)
- Validation-week score of the final run **[TBD]**; test score is computed by the course after the test week
- Limits: differences are within seed noise (0.5–1 point); whether push is scored is unknown; GloVe text features were not ablated; a graph transformer (RelGT) was surveyed but not implemented; reproduction on another GPU may change a few 0/1 entries

## 16. Conclusion and references
- Three things helped: self-supervised pretraining, relative time encoding, recent per-user statistics
- More relations, augmentation and extra losses did not help; RDL's main value here is removing hand-made joins
- Honest summary: gains are modest; simple baselines are competitive
- References: GH Archive; Fey et al., Relational Deep Learning (ICML 2024); Robinson et al., RelBench (NeurIPS 2024); RelGNN, RelGT, Relational Transformer (as surveyed, see `reports/rdl/survey_rdl.md`)
- Code: GitHub URL **[TBD]**

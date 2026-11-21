# Submission rules notes

Checked 2026-11-20 (Taiwan time).

## Official rules document: NOT readable

The Google Doc "[Revision] Big Data Course - Assignment 2" requires a Google login. WebFetch only
returned the sign-in page, so nothing in it was read. Everything below comes from the files in this
project (README.md, SETUP.md, SCHEMA.md, scripts/evaluate.py, scripts/preprocess.py,
scripts/download_gharchive.py). Items marked UNKNOWN must be checked in the Doc by a person.

## What the project files say

| Topic | Source | Rule |
|---|---|---|
| Output file | README "4. Test users and output" | `output_{studentID}.csv`, one row per user in `submission/test_input.csv` (4,465), header exactly `UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push`, values 0 or 1, no empty cells. A file with a format error is not scored. |
| Format check | README, evaluate.py | `python scripts/evaluate.py --pred output_{studentID}.csv --input submission/test_input.csv` |
| Test week | README | Taiwan time 2026-11-22 00:00 to 11-29 00:00 (UTC 11-21 16:00 to 11-28 16:00, end excluded) |
| Missing data | README "Missing data" | If GH Archive lacks an activity during the test week, that activity is not scored. |
| Score | README "Scoring" | F1 per activity, Score = (macro-F1 + micro-F1) / 2 |
| Preprocessing | README sections 1-2 | "You are required to run our provided preprocessing script before using the data for your model." Raw hourly files come from GH Archive into your own folder. |
| Row-time rule | README section 3 | For a prediction at time t use only rows whose row time is before t. Do not filter PullRequests, Issues, PRReviews by `created_at`/`submitted_at`. |
| Data range | README section 1 | You choose the period to download. Nothing in the README limits it, except that rows must be before the test start. |
| Data after the local end | SETUP.md "資料流程" | Local data is UTC 2026-10-01 to 2026-11-18 00:00 (excluded) = Taiwan 11/18 08:00. SETUP says the final prediction should update the data to before the test start, and if the whole UTC 11/21 is downloaded the model must still cut at 2026-11-21 16:00 UTC. |
| Downloader limit | scripts/download_gharchive.py | `--end` is an inclusive UTC date and must be a completed UTC day. So with the current script the newest usable end is the last completed UTC day: 11/20 (available from Taiwan 11/21 08:00), leaving a 16 h gap before the cutoff. Hours 11/21 00:00 to 15:00 UTC cannot be fetched with `--end` before 11/22 00:00 UTC. |

## UNKNOWN (only in the Google Doc, must be checked by a person)

- The submission deadline and whether it falls before or after the test week starts.
- Whether the TAs rerun the code and must reproduce the submitted CSV exactly (the task brief says they
  rerun the provided preprocessing; the README only says preprocessing is required). Exact versus
  approximate reproduction is unknown.
- Whether model code, a report, or a README "Data Source" section must be submitted with the CSV, and the
  file format of the submission (the task brief mentions output_{studentID}.csv only).
- Any seed or determinism requirement.
- Any restriction on features or on external data.

## Conflicts with what we have done so far

1. The report draft (reports/question234_draft.md, Q3 item 5) says all cutoffs sit on Taiwan Sundays.
   The final pipeline uses the latest Taiwan-midnight week that ends inside the data (11/11 with the
   current data, a Wednesday), because 11/15 (Sunday) would end after the data. Update the sentence.
2. All earlier experiments predicted at a cutoff with zero data gap. The real test has a gap of
   88 h with the current files (16 h after downloading through 11/20). The final pipeline trains and
   validates with the same gap, so earlier scores (about 38-39 percent) are not directly comparable.
3. The draft Data Source text says "2026-10-01 to 2026-11-18 UTC". It must be updated if more files are
   downloaded.
4. The final model needs the RelBench clone and venv under data/model/relbench_public (git-ignored
   `data/`). If the TAs must rerun the training, those must be documented or shipped.

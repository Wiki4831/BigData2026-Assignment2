# Assignment 2 準備與操作

## 環境

直接使用現有 conda 環境 `bigdata`，無須重新安裝：

```bash
conda activate bigdata
cd /home/asiadragon/geolife/bigdata_asg/BigData2026-Assignment2
bash run.sh env
```

已驗證 Python 3.11.17、NumPy 2.4.6、pandas 3.0.6、PyArrow 26.0.0。
JupyterLab 與 ipykernel 也已安裝；VS Code Notebook 可選擇
`/home/asiadragon/miniconda3/envs/bigdata/bin/python`。
`run.sh` 直接使用這個 Python，因此未 activate 也可執行。
若改用其他環境，可用 `ASG2_PYTHON=/path/to/python bash run.sh env`。

`requirements-setup.txt` 記錄此次驗證的核心套件版本。
`environment.yml` 提供另一台機器建立環境的最小設定：

```bash
conda env create -f environment.yml
ASG2_PYTHON=/path/to/bigdata-asg2/bin/python bash run.sh env
```

官方前處理及評分只需要 CPU。已在沙箱外確認 RTX 4090（24 GB）、
NVIDIA 驅動 580.178.04 與 PyTorch 2.11.0+cu128 可使用 CUDA。
受限沙箱內的 `nvidia-smi`／`torch.cuda.is_available()` 會顯示失敗／False，
不代表主機驅動故障。主模型使用目前 `bigdata` 環境已有的 PyTorch。

## 資料流程

目前資料已更新至 **UTC 2026-10-01 00:00 至 2026-11-18 00:00（不含）**，
即 10/01–11/17 的全部 1,152 個小時檔。
台灣時間對應 10/01 08:00 至 11/18 08:00（不含）。
11/18 的部分時段未混入本次資料，以維持完整 UTC 日的每日分析。
日期範圍依 [GH Archive 官方小時檔命名規則](https://www.gharchive.org/) 使用 UTC。

```bash
# 先檢視範圍，不下載
bash run.sh download --dry-run

# 重現本次資料範圍；重跑會驗證並略過已有的完整檔案
bash run.sh download --start 2026-10-01 --end 2026-11-17

# 不指定 --end 時，預設為最新已結束的 UTC 日
# bash run.sh download

# 執行老師提供的原版腳本，輸出 8 張匿名化表格
bash run.sh preprocess

# 檢查欄位、Arrow 型別、主鍵、外鍵、時間排序及 cohort
bash run.sh check --report data/verification.json
```

下載使用 6 個連線，檢查回傳檔案大小與 gzip 完整性，失敗會重試。
未完成的檔案使用 `.part` 副檔名，不會被官方前處理讀入。
404 與重試失敗會記錄在 manifest 並以非零狀態結束；此時先補抓，
確認完整性後再前處理。不要在下載尚未完成時建置正式表格。
老師的 `scripts/preprocess.py`、`scripts/evaluate.py`、cohort 與 submission 均保留原樣。

| 路徑 | 用途 |
|---|---|
| `data/raw/gharchive/` | 未修改的 GH Archive 原始 `.json.gz` |
| `data/download_manifest.json` | 各小時下載狀態、檔案大小、缺檔或錯誤 |
| `data/processed/db/` | 官方生成的 8 張 Parquet 表 |
| `data/verification.json` | 正式表格檢查及 cohort／測試使用者覆蓋率 |
| `data/environment_snapshot.json` | 本機已驗證的 Python 和套件版本 |
| `notebooks/01_initial_analysis.ipynb` | 已執行、附輸出與圖表的中文初探 Notebook |
| `data/analysis/` | Notebook 產生的 CSV、PNG 與 `eda_summary.json` |
| `data/snapshots/before_update_ending_2026-11-13/` | 更新前、截止 UTC 11/13 00:00 的表格與建置紀錄 |
| `data/smoke/` | 第一個真實小時檔的流程驗證，非訓練資料集 |
| `logs/` | 完整前處理與檢查紀錄 |

`data/` 與 `logs/` 已列入 `.gitignore`，避免將大量資料提交到 Git。
完整資料的實際大小、各表筆數與驗證結果見本檔案下方的建置結果。

## 作業規則重點

- 預測 6 種活動，測試清單共 4,465 位使用者；cohort 共 9,000 位。
- 測試週為 **台灣時間 2026/11/22 00:00–11/29 00:00**，
  對應 **UTC 11/21 16:00–11/28 16:00**，結束時間不含。
- 使用者對接應使用 `test_input.UserID = Users.github_user_id`；
  `Users.user_id` 是該次建表的列號，處理不同期間後可能改變。
- 所有原始資料須先經官方 `preprocess.py`，模型使用匿名化表格。
- 時間切分依 README／SCHEMA 的 **row time**：Users、Repositories、
  PullRequests、Issues、PRReviews 用 `first_seen_at`；
  IssueComments、PRReviewComments、Pushes 用 `created_at`。
  預測時間 t 只允許 row time < t 的列。
- PullRequests、Issues、PRReviews 不可只依 `created_at`／`submitted_at` 判斷可見性。
  `merged_at`、`closed_at`、`dismissed_at`、`merged_by_id` 等事後才知道的資訊，
  在歷史驗證時也須按預測時間處理，避免洩漏未來。
- 官方分數為 `(macro-F1 + micro-F1) / 2`。
- 目前截止時間之後、測試開始前仍會有新的歷史資料；本次選定範圍可用於起步訓練與驗證，
  最終預測前應更新至測試起點之前。若下載整個 UTC 11/21，
  建模仍必須以 `2026-11-21 16:00:00 UTC` 截斷。

```bash
# 檢查提交格式與 UserID 是否符合正式測試清單
bash run.sh evaluate --pred output_學號.csv --input submission/test_input.csv

# 自行切出歷史驗證週之後，評分預測和答案
bash run.sh evaluate --pred validation_pred.csv --truth validation_truth.csv
```

官方 `sample_submission.csv` 是隨機預測的格式示例，不是訓練完成的模型結果。

## 初步分析 Notebook

直接在 VS Code 開啟 `notebooks/01_initial_analysis.ipynb`，選擇 conda `bigdata` kernel。
檔案已保留完整執行結果；更新資料後可重新 Run All。或在作業根目錄執行：

```bash
conda activate bigdata
jupyter lab notebooks/01_initial_analysis.ipynb
```

Notebook 會自動從目前目錄或上層尋找專案，不需要修改絕對路徑。內容包括：

- 小時檔覆蓋率與資料截止時間；8 張表的概況與可切換的內容預覽。
- 欄位型別、缺值、主鍵與外鍵檢查。
- cohort／測試使用者覆蓋率，以及沒有可歸屬活動記錄的使用者。
- `first_seen_at` 與建立／提交時間之間的延遲，及歷史可見性限制。
- 六種活動的表格推估、每日趨勢、完整歷史週活躍率。
- 每日零量診斷、使用者活躍度與近期性、User／Repo 關聯查詢。
- 預測時間前的七天歷史計數範例，以及後續時間驗證的入口。

這些活動統計是 **table-derived proxy**，不是官方答案；例如 PR 開啟資訊可能缺失，
或從較晚事件才補得。Notebook 同時保留活動時間和可見時間，不把新表列直接當成開啟事件。
`data/analysis/test_user_history.csv` 區分「沒有 Users 列」和「沒有定義中的活動 proxy」。
所有圖表與 CSV 可在 `data/analysis/` 取得，摘要為 `data/analysis/eda_summary.json`。

已用現有 `bigdata` 環境執行全部 14 個程式儲存格，保留輸出，並產生 17 份 CSV 與 5 張圖。
初探發現全期間有 12 天的 push proxy 為零；完整歷史週的觀察活躍率也有明顯變動。
這些現象已列入診斷，尚不能直接歸因於使用者行為或來源缺漏。

## 主模型

[資料與時間切分](scripts/model_data.py)、[異質 GraphSAGE](scripts/gnn_model.py)、[訓練與推論入口](scripts/train_model.py) 可依這個順序閱讀。英文註解標明圖索引、缺失歷史、時間可見性及共用向量的處理。 [模型草稿](reports/model_draft.md)

```bash
bash run.sh train --device auto --refit
bash run.sh predict --device auto --pred data/model/predictions.csv
bash run.sh evaluate --pred data/model/predictions.csv --input submission/test_input.csv
```

`train` 預設用 11/1 那週選模型和閾值，11/8 那週做獨立檢查。各回看窗口都附帶覆蓋率，所以 10/4 和 10/11 也能作為訓練週。`--refit` 在檢查後固定 epoch 數和閾值，再用所有符合條件的測試前標籤週重訓。Checkpoint 預設寫到 `data/model/asg2.pt`。可用 `--epochs 2 --device cpu` 短跑確認流程。

更新資料時重新執行官方前處理，讓 `--db` 指向新的 Parquet，`--manifest` 指向與它對應的小時檔清單。正式預測只讀台灣時間 11/22 00:00 以前已可見的資料。如果已知有個別活動收錄中斷，訓練與推論可共用 `--missing-activity-csv`。CSV 欄名為 `activity,start,end`，起訖為整點，結束時間排除。請使用帶時區的時間，例如 `2026-11-10T04:00:00Z`。這份可選清單用來區分未知標籤和數值零，也讓歷史特徵帶有各活動的覆蓋比例。

### 按日建圖的實驗

`--daily-graphs` 把同一個週切點前七天分成七張圖，預測目標仍為接下來七天的六項活動。每天由當天首次可見的活動列出發，保留關聯到的舊 PR、issue 等資料，並以共用 GraphSAGE 編碼。模型按日期順序合併七個向量，也保留原本的個人活動統計。這個旗標訓練時預設存為 `data/model/daily_asg2.pt`。推論依 checkpoint 自動選正確的建圖方式。

```bash
bash run.sh train --daily-graphs
bash run.sh predict --checkpoint data/model/daily_asg2.pt --pred data/model/daily_predictions.csv
```

### 用逐日 loss 訓練，按週評估

`--daily-loss` 每天取一個訓練切點，每個切點展開七步。模型用當時的歷史圖算初始向量，以 GRU 根據自己的每日預測更新向量，再預測下一天。Loss 是七天所有可用的下一天二元標籤的平均 BCE。每個訓練切點的七天答案都要在驗證切點前結束。逐日標籤依事件發生時間分天，並沿用週末的標籤可見截止，使七天的標籤取 OR 後和原本週標籤一致。

驗證和留出週都從週開始前一路預測七天。每種活動用一個共用閾值，任一天過閾值就輸出週 `1`。閾值及最佳 epoch 以整週作業分數選擇。這個模式可使用累積圖，也能搭配 `--daily-graphs` 組合歷史圖。訓練預設將此模式存成 `data/model/next_day_asg2.pt`，合用日圖時為 `data/model/next_day_daily_asg2.pt`。

```bash
bash run.sh train --daily-loss
bash run.sh predict --checkpoint data/model/next_day_asg2.pt --pred data/model/next_day_predictions.csv
```

### 篩選 GNN 的活動歷史

`--graph-history-days 7` 只用切點前最近七天首次可見的活動列作為建圖起點，再沿外鍵保留它們需要的較早 PR、issue 或 repo。使用者統計特徵和冷啟動條件沿用原本完整歷史，方便單獨比較 GNN 鄰居範圍。Checkpoint 會記錄圖窗口，推論時沿用同一種建圖。訓練時可搭配 `--daily-loss`。`--daily-graphs` 是另一種歷史建圖方式，兩種建圖選項擇一。

```bash
bash run.sh train --daily-loss --graph-history-days 7
bash run.sh predict --checkpoint data/model/next_day_recent_7d_asg2.pt --pred data/model/next_day_recent_7d_predictions.csv
```

## 最新建置結果

2026-11-18 完成更新：

- 新增 11/13–11/17 的 120 小時檔，沿用原有 1,032 個檔案，合計 1,152 / 1,152，缺檔 0。
- 原始壓縮資料 9,821,721,125 bytes（約 9.15 GiB）；8 張 Parquet 合計約 17.46 MiB。
- 官方前處理保留 69,085 筆 cohort 活動；欄位型別、主鍵、外鍵與時間檢查全部通過。
- 舊表格與舊建置摘要保留在 `data/snapshots/before_update_ending_2026-11-13/`。
- 正式文件、前處理腳本、評分腳本、cohort 與 submission 均未修改。

| 表 | 本次筆數 | 原 11/12 版本筆數 |
|---|---:|---:|
| Users | 5,964 | 5,777 |
| Repositories | 12,733 | 12,326 |
| PullRequests | 14,554 | 13,394 |
| Issues | 9,279 | 8,557 |
| IssueComments | 3,673 | 3,346 |
| PRReviews | 3,563 | 3,307 |
| PRReviewComments | 1,797 | 1,705 |
| Pushes | 43,545 | 42,831 |

本次 Users 表涵蓋 4,298 / 9,000 位 cohort 使用者，以及 3,732 / 4,465 位測試使用者。
未出現的測試使用者由 757 位降為 **733 位**，已更新
`data/test_users_without_observed_history.csv`。預測須以完整測試清單做 left join，避免漏交。

正式表格檢查為 `data/verification.json`，建置摘要為 `data/build_summary.json`。
執行紀錄是 `logs/download_2026-10-01_to_2026-11-17.log`、
`logs/preprocess_2026-10-01_to_2026-11-17.log` 與
`logs/verification_2026-10-01_to_2026-11-17.log`。

## 首次建置紀錄（歷史版本）

2026-11-13 完成：

- 1,032 / 1,032 小時檔下載成功，缺檔 0，下載失敗 0。
- 每個檔案均通過回傳大小及 gzip 完整性檢查；完整時段與實際檔案大小也已再次核對。
- 原始壓縮資料 9,204,825,727 bytes（8.57 GiB）；8 張 Parquet 合計約 16.36 MiB。
- 官方前處理保留 66,398 筆 cohort 活動。
- 8 張表的欄位、Arrow 型別、連續主鍵、外鍵、時間排序及 cohort 標記檢查全部通過。
- 官方範例提交檔通過 4,465 位測試使用者的格式檢查。
- 官方原有文件、腳本、cohort 與 submission 均無修改，來源版本及檔案雜湊記在環境快照。
- 已實際在 RTX 4090 上完成小型 CUDA tensor 運算，確認既有 PyTorch 可用。

| 表 | 筆數 |
|---|---:|
| Users | 5,777 |
| Repositories | 12,326 |
| PullRequests | 13,394 |
| Issues | 8,557 |
| IssueComments | 3,346 |
| PRReviews | 3,307 |
| PRReviewComments | 1,705 |
| Pushes | 42,831 |

本段資料的 Users 表涵蓋 4,227 / 9,000 位 cohort 使用者，以及
3,708 / 4,465 位正式測試使用者。其餘 **757 位測試使用者在這段資料的 Users 表未出現**，
清單為 `data/test_users_without_observed_history.csv`。
這不代表未來不會活動；建立特徵與預測時，應以完整 `submission/test_input.csv`
為主表做 left join，為缺乏歷史的使用者設定預設特徵／預測，避免漏交使用者。
原始小時檔完整也不保證 GH Archive 收錄了每一個 GitHub 活動；官方已提醒部分來源時段可能不完整。

本次完成資料與環境準備；尚未訓練模型或建立正式預測。正式檢查結果為
`data/verification.json`，完整執行紀錄在 `logs/`。

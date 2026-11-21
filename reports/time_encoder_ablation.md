# ASG2 獨立時間編碼對照

## 目前模型如何用時間

GNN 的節點特徵包含資料列距離切點的天數、活動距離切點的天數、事件時間是否已知。時間差用 `log1p` 縮放。Review 額外有 PR 開啟到提交 review 的間隔，使用者也有多個回看窗口的活動次數和近期性。這些特徵和其他欄位一起投影成 64 維。GRU 的第幾天向量則用來表達未來七步中的預測位置。 [資料特徵](../scripts/model_data.py#L195) · [逐日模型](../scripts/gnn_model.py#L97)

RelBench 範例有一個獨立的時間分支。它用切點減節點 row time 得到相對時間，以天為單位做正弦和餘弦的位置編碼，再依節點類型線性轉換，加到節點欄位的向量上，最後送入 GNN。作業 README 只說明參考模型有使用相對時間。 [RelBench 時間編碼](https://github.com/stanford-star/relbench/blob/main/relbench/modeling/nn.py) · [RelBench 模型範例](https://github.com/stanford-star/relbench/blob/main/examples/model.py) · [作業參考基準](../README.md#4-test-users-and-output)

## 同一資料切分的對照

在現有逐日 loss 模型上，保留原本時間欄位，再加入一條獨立的時間分支。從節點的 `log1p(row_age_days)` 還原相對天數，以和 PyG 預設位置編碼同樣的 32 個頻率產生 64 維正弦及餘弦特徵。每個節點類型有獨立的 64 維線性層，結果加到節點向量。兩個版本採相同的 23 個歷史日切點、種子 42、累積圖及逐日 BCE，皆依 11/1 驗證週選 epoch 和閾值，再評分 11/8 留出週。原有層的初始隨機參數保持相同。 [原本訓練方式](../SETUP.md#用逐日-loss-訓練按週評估) · [PyG 位置編碼原始碼](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/encoding.html#PositionalEncoding)

| 逐日模型 | 11/1–11/8 驗證總分 | 11/8–11/15 留出總分 | 留出 macro-F1 | 留出 micro-F1 |
|---|---:|---:|---:|---:|
| 原本時間欄位 | 37.94% | 34.61% | 34.39% | 34.83% |
| 加獨立時間分支 | 37.06% | 34.60% | 34.31% | 34.89% |

單一種子的留出分數相近，驗證分數下降約 0.89 個百分點。這次只改變節點時間表示，聚合仍對同類鄰居等權取平均。

## 參考週差異

官方 54.82% 是 9/27–10/4 的參考分數。同一份官方表中，照抄上週為 49.61%。目前 Parquet 從 UTC 10/1 開始，無法用同一份資料重做該參考週。在資料可涵蓋完整一週回看和標籤週的本機回測中，照抄上週在 10/11–10/18 得 45.62%，在 11/8–11/15 得 32.85%。這組固定規則在不同週就有顯著數值變動，也受到 push 收錄變化影響。 [作業參考基準](../README.md#4-test-users-and-output) · [資料建置與期間](../SETUP.md#最新建置結果) · [收錄抽查](../notebooks/02_user_activity_timing.ipynb)

## 同一週的 RelBench 元件對照

RelBench 的公版 entity 範例用 PyTorch Frame 編碼欄位，加入依節點類型的時間向量，然後對每批目標實體依切點抽取時間鄰居。範例預設 `aggr=sum`，GraphSAGE 每層使用 LayerNorm，最後直接輸出任務標籤。ASG2 的正式標籤和六項 F1 要另接任務定義。 [RelBench entity 訓練範例](https://github.com/stanford-star/relbench/blob/main/examples/gnn_entity.py) · [欄位編碼與聚合](https://github.com/stanford-star/relbench/blob/main/relbench/modeling/nn.py)

先前的元件對照沿用原本八張表、欄位編碼、全圖、共用向量及 64 維，將直接週預測 GNN 改成獨立時間向量、同關係訊息求和、每關係的來源及目的轉換、每層 LayerNorm。以同樣四個歷史週和種子 42 訓練，11/1 選 epoch 和閾值，11/8 評分。 [現有欄位及建圖](../scripts/model_data.py) · [模型選擇程式](../scripts/train_model.py)

| 方法 | 11/1–11/8 驗證總分 | 11/8–11/15 留出總分 |
|---|---:|---:|
| 原本的直接週預測 GNN | 37.03% | 33.49% |
| 直接週預測 GNN，加入上述公版元件 | 39.21% | 35.18% |
| 逐日 BCE 與七步更新（23 個日切點） | 37.94% | 34.61% |

直接週預測的成對比較提升 1.69 個百分點。模型同時變更時間編碼、聚合及正規化，結果限於本週和單一種子。

## 完整 RelBench 公版回測

使用 RelBench 3.1.0 的原版 `Model`、PK–FK 建圖、PyTorch Frame 欄位編碼、GloVe 文字編碼、時間鄰居取樣、相對時間編碼與兩層異質 GraphSAGE。沿用公版的 128 維、取樣上限 128／64、batch size 512、Adam 0.005 和 10 個 epoch。GloVe 處理的是作業資料中已遮罩的文字。 [公版訓練設定](https://github.com/stanford-star/relbench/blob/5bec345fa6a424641152d8b45d91535d6a9750c4/examples/gnn_entity.py) · [公版模型](https://github.com/stanford-star/relbench/blob/5bec345fa6a424641152d8b45d91535d6a9750c4/examples/model.py)

ASG2 接合程式改為輸出六個週活動標籤，以有標籤的欄位算二元交叉熵。9,000 位使用者都納入評估，切點前沒有可用關係的使用者共用一個無邊節點。欄位編碼統計在 11/8 切點前的共用圖建立，每個訓練及驗證樣本依自己的切點取樣。拿掉事後才知道的欄位及晚於 PR 資料列出現時間才可確定的作者關係。重新執行前修正了目標切點的時間單位，並檢查六個週切點共 54,000 筆使用者樣本的順序及取樣節點時間。 [ASG2 接合程式](../scripts/relbench_asg2.py)

訓練週從 10/4、10/11、10/18、10/25 開始，週切點為台灣時間。用 11/1–11/8 驗證週的六類平均 ROC-AUC 選出第 2 個 epoch（89.76%），也用該驗證週校準六個 F1 閾值。種子為 42，RTX 4090 上的 10 個 epoch 含驗證共約 65 秒，另有建圖及文字編碼時間。

| 方法 | 11/1–11/8 驗證總分 | 11/8–11/15 留出 macro-F1 | 留出 micro-F1 | 留出總分 |
|---|---:|---:|---:|---:|
| 照抄上週 | — | 32.71% | 32.98% | 32.85% |
| 原本的直接週預測 GNN | 37.03% | — | — | 33.49% |
| 逐日 BCE 與七步更新 | 37.94% | 34.39% | 34.83% | 34.61% |
| 直接週預測 GNN，加入部分公版元件 | 39.21% | — | — | 35.18% |
| 完整 RelBench 公版 | 38.48% | 35.72% | 36.25% | **35.98%** |

公版留出週的六類 F1 依序為 `open_pr` 37.79%、`open_issue` 42.99%、`issue_comment` 36.25%、`pr_review` 40.64%、`pr_review_comment` 22.87%、`push` 33.79%。4,118 位有可用歷史關係的使用者總分為 36.67%，其餘 4,882 位共用冷啟動預測，總分為 0%。官方 54.82% 的參考週是 9/27–10/4，當前資料從 UTC 10/1 開始。 [作業參考分數](../README.md#4-test-users-and-output)

在專案根目錄使用已安裝的獨立環境和快取重跑。

```bash
HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1 \
  data/model/relbench_public/venv/bin/python scripts/relbench_asg2.py --device cuda
```

輸出在 `data/model/relbench_public/metrics.json` 和 `metrics.pt`。

## RelBench 圖編碼與逐日 GRU

`--daily-loss` 使用同一張八表關聯圖。每個預測切點只取樣當時可見的鄰居，經 PyTorch Frame、相對時間編碼和兩層異質 GraphSAGE 得到 128 維使用者向量。用線性層投影成 64 維修正向量，與可訓練的共用使用者向量相加，作為原本 GRU 的初始狀態。沒有歷史關係的使用者，其個人修正向量為零。 [RelBench 圖編碼接合](../scripts/relbench_asg2.py) · [原本的七步 GRU](../scripts/gnn_model.py)

每個切點展開未來七天。每日輸出六個活動機率，GRU 用當日預測機率和日期位置更新隱狀態。訓練以每天的六個標籤計算 BCE，梯度經 GRU 回傳到圖編碼器。評估時對各活動使用驗證週校準的閾值，七天任一天達到閾值即判為該週有活動。

沿用 23 個每日訓練切點、64 維 GRU、AdamW 0.001 和相同的週驗證與留出切分。公版時間取樣以 512 筆使用者樣本為一批，圖編碼器和 GRU 一起從頭訓練。已檢查 207,000 筆使用者切點的時間、標籤對齊和歷史取樣。首批逐日 BCE 在兩層 `Pushes → Users` 的鄰居權重產生梯度，L2 分別為 0.00104 和 0.00379，更新後權重改變的 L2 分別為 0.12425 和 0.12120。GRU 的輸入權重也有梯度及更新。舊逐日模型同樣已驗證兩層 `Pushes → Users` 的關係轉換可收到梯度。圖的外鍵決定資料列連線，模型學習各種關係的鄰居訊息如何轉換。

種子 42 的驗證總分為 38.97%，選出第 1 個 epoch。RTX 4090 上完成 7 個 epoch 後達到早停條件，訓練連同逐週驗證約 4 分 13 秒，另計建圖時間。

| 方法 | 11/8–11/15 留出 macro-F1 | 留出 micro-F1 | 留出總分 |
|---|---:|---:|---:|
| 原本的逐日 GNN＋GRU | 34.39% | 34.83% | 34.61% |
| 完整 RelBench 直接預測一週 | 35.72% | 36.25% | 35.98% |
| RelBench 圖編碼＋逐日 GRU | 35.97% | 36.48% | **36.22%** |

有可用歷史關係的 4,118 位使用者得 36.93%，其餘 4,882 位得 0%。新版比原逐日模型高 1.61 個百分點。此對照使用一個留出週和一個種子，圖編碼與訓練批次做法都隨公版改動。

```bash
HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1 \
  data/model/relbench_public/venv/bin/python scripts/relbench_asg2.py --device cuda --daily-loss
```

此模式把分數和模型權重寫入 `data/model/relbench_public/daily_gru_metrics.json` 和 `daily_gru_metrics.pt`。

## 先預訓練圖編碼器，再訓練 GRU

第一階段從歷史事件自動產生下一個 24 小時的六項二元標籤。圖只取樣各切點前的資料。以 `共用向量 + RelBench 個人修正` 接線性預測頭，從頭聯合訓練 PyTorch Frame 欄位編碼、相對時間編碼、GraphSAGE、64 維投影、共用向量和預測頭。預訓練使用 10/3–10/31 的 29 個每日切點及 9,000 位使用者，共 261,000 筆。最後一批標籤涵蓋至 11/1 驗證切點，欄位型態、統計與類別詞彙也只從驗證切點前的資料建立。後續資料由同一套欄位轉換器編碼。

第二階段沿用預訓練的共用向量與預測頭作為初值，凍結 RelBench 編碼器和 64 維投影，訓練原本的七步 GRU、共用向量、每日位置向量與預測頭。沿用從頭訓練版的 23 個 GRU 訓練切點和逐日 BCE。由 11/1–11/8 驗證週選 epoch 及閾值，最後評估 11/8–11/15 的週預測。 [兩階段訓練](../scripts/relbench_asg2.py)

RTX 4090 上的 3 個預訓練 epoch 約 2 分 03 秒，單日 BCE 從 0.04830 降到 0.04474。GRU 訓練 10 個 epoch 後早停，含驗證約 3 分 48 秒，最佳是第 4 個 epoch。已比對兩階段 checkpoint，圖編碼器與投影層的 691 個張量完全相同，共用向量在第二階段有更新。

| 方法 | 驗證週總分 | 留出週 macro-F1 | 留出週 micro-F1 | 留出週總分 |
|---|---:|---:|---:|---:|
| RelBench＋GRU 從頭一起訓練 | 38.97% | 35.97% | 36.48% | 36.22% |
| 預訓練 RelBench、凍結編碼器後訓練 GRU | 39.80% | 37.50% | 38.18% | **37.84%** |

有歷史關係的 4,118 人得 38.67%，沒有歷史關係的 4,882 人得 0%。`issue_comment` 的留出 F1 從 32.09% 升至 40.91%，`pr_review` 從 43.06% 降至 40.42%。兩階段版本和從頭訓練版本的欄位統計取樣範圍也有差異。比較採種子 42 和一個留出週。

```bash
HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1 \
  data/model/relbench_public/venv/bin/python scripts/relbench_asg2.py \
  --device cuda --daily-loss --pretrain-epochs 3
```

分數、預訓練權重和最後的 GRU 權重分別保存在 `data/model/relbench_public/pretrained_gru_metrics.json`、`pretrained_gru_metrics.encoder.pt` 和 `pretrained_gru_metrics.pt`。

## 2、3、5-hop 比較

沿用兩階段訓練，讓 GraphSAGE 層數和時間取樣深度一起增加。各跳鄰居上限依原有規則每跳減半，分別為 `[128, 64]`、`[128, 64, 32]` 和 `[128, 64, 32, 16, 8]`。3-hop 和 5-hop 各自從頭預訓練 3 個 epoch，再凍結圖編碼器訓練 GRU。訓練、驗證與留出切分、batch size 512 和種子 42 都採同一設定。

在 9,000 位使用者的留出週取樣檢查中，三種設定都有取到最遠一跳，所有取樣節點都符合各自的時間切點。2-hop、3-hop 和 5-hop 的最大單批取樣節點數分別為 5,922、6,873 和 8,551。4,118 位有歷史的使用者中，取樣子圖至少包含一個其他使用者節點的人數分別為 199、556 和 797。

| 深度 | 最佳 GRU epoch | 11/1–11/8 驗證總分 | 11/8–11/15 留出 macro-F1 | 留出 micro-F1 | 留出總分 |
|---|---:|---:|---:|---:|---:|
| 2-hop | 4 | 39.80% | 37.50% | 38.18% | 37.84% |
| 3-hop | 6 | 38.70% | 37.72% | 38.40% | **38.06%** |
| 5-hop | 8 | 39.08% | 36.85% | 37.56% | 37.21% |

3-hop 和 5-hop 在預訓練首批的每一層都有非零梯度及權重更新。比對預訓練和最終 checkpoint，兩組凍結期間的圖編碼器與投影層參數完全相同。3-hop 的留出分數比 2-hop 高 0.22 個百分點，5-hop 比 2-hop 低 0.63 個百分點，三組無歷史關係的使用者得分皆為 0%。比較採單一種子與單一留出週。

```bash
for hops in 3 5; do
  HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1 \
    data/model/relbench_public/venv/bin/python scripts/relbench_asg2.py \
    --device cuda --daily-loss --pretrain-epochs 3 --hops "$hops"
done
```

兩次實驗的權重和分數各自寫入 `data/model/relbench_public/` 中帶有 `_hop3` 或 `_hop5` 的檔案。

## 原版 3-hop 預訓練的收斂檢查

從頭重跑原版 3-hop、29 個訓練切點和 AdamW 0.001。每輪以原版的六類下一天 BCE 訓練，另用 11/1–11/7 七個切點的 63,000 筆使用者日計算相同的 BCE。驗證取樣在每輪使用同一個種子，欄位型態、統計和類別詞彙僅以訓練期資料建立。

連續六輪驗證 BCE 未比上一次顯著改善點降低超過 0.0001 即早停。保存驗證 BCE 最低的權重，另記錄每輪訓練 BCE 以區分訓練 loss 持續下降與驗證停滯。

| Epoch | 訓練 BCE | 驗證 BCE |
|---:|---:|---:|
| 1 | 0.048462 | 0.025491 |
| 2 | 0.045405 | **0.025088** |
| 3 | 0.044567 | 0.025449 |
| 4 | 0.044072 | 0.025230 |
| 5 | 0.043731 | 0.025145 |
| 6 | 0.043395 | 0.025609 |
| 7 | 0.043106 | 0.025532 |
| 8 | 0.042869 | 0.025361 |

第 8 輪觸發早停，保存第 2 輪的最佳驗證權重。RTX 4090 的 8 輪訓練、驗證合計 351 秒。先前保存的 3-epoch 編碼器以相同驗證切點和取樣種子在 CPU 重新評估，BCE 為 0.025511。這個比較評估隔天的預訓練目標，整週 GRU 分數需另外測量。

```bash
HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1 \
  data/model/relbench_public/venv/bin/python scripts/relbench_asg2.py \
  --device cuda --hops 3 --pretrain-only
```

結果、最佳權重及 loss 圖在 `data/model/relbench_public/` 的 `pretrain_convergence_hop3.json`、`pretrain_convergence_hop3.encoder.pt` 和 `pretrain_convergence_hop3.png`。

## 遮蔽歷史活動並猜類型

預訓練樣本以「使用者 × 台灣日期 × 已觀測活動類型」分組。一個日期按事件和資料列都可見的時間歸屬。同一組的事件一起遮蔽，並移除所有正反向邊。被遮住的 PR 所對應的 issue 討論列也一併遮蔽。將樣本分為八組，各組在自己的遮蔽圖上取樣，同一人同一天的不同活動分到不同遮蔽組。保留遮蔽後仍有可用歷史的樣本，讓 3-hop RelBench 用剩餘歷史猜六種活動之一，並以類型頻率的倒數加權交叉熵。

訓練有 19,948 筆可用樣本，驗證有 2,014 筆。用 11/1–11/8 的遮蔽驗證樣本選最小的類別平衡交叉熵。取樣檢查確認了全部訓練和驗證樣本的目標對齊、切點和被遮掉的資料列。轉入 GRU 時保留預訓練圖編碼器、64 維投影和共用向量，凍結圖編碼器及投影，重新初始化預測頭。GRU 仍在完整 cohort 用逐日 BCE 訓練，採原有 23 個切點、驗證及留出週。

初版隨機分組時，有 6.16% 的候選訓練樣本與同人同日的其他類型落在同一遮蔽組，會產生相同輸入對應多個答案。修正版消除了這些碰撞。遮蔽預訓練在第 11 輪早停，選第 5 輪。最佳驗證交叉熵為 1.33605，macro recall 為 51.34%。各類型的召回率依序為 `open_pr` 45.95%、`open_issue` 66.28%、`issue_comment` 83.52%、`pr_review` 36.91%、`pr_review_comment` 57.81% 及 `push` 17.55%。固定猜任一類的 macro recall 為 16.67%。預訓練和逐輪驗證合計約 52 秒，另計建圖時間。

| 3-hop 預訓練方式 | 11/1–11/8 驗證總分 | 11/8–11/15 留出 macro-F1 | 留出 micro-F1 | 留出總分 |
|---|---:|---:|---:|---:|
| 原版隔天 BCE，3 個預訓練 epoch | 38.70% | 37.72% | 38.40% | **38.06%** |
| 遮蔽活動並猜類型 | 37.56% | 34.86% | 35.42% | 35.14% |

修正版 GRU 於第 13 輪早停，選第 7 輪。留出總分較原版低 2.92 個百分點。有歷史組得 35.88%，無歷史組得 0%。已比對兩階段 checkpoint，凍結的 821 個圖編碼器與投影張量完全相同。此對照同時改變預訓練目標、抽樣與預測頭初始化，採一個種子及一個留出週。初版的留出分數為 35.27%，其結果另行保存。

```bash
HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1 \
  data/model/relbench_public/venv/bin/python scripts/relbench_asg2.py \
  --device cuda --hops 3 --daily-loss --masked-type-pretrain --pretrain-min-delta 0.001
```

分數和權重寫入 `data/model/relbench_public/` 中以 `masked_type_gru_metrics_hop3` 為檔名開頭的檔案。

## 遮蔽 Review 欄位並保留連線

以一筆可見的 Review 為重建樣本，把該筆 Review 的所有欄位從欄位編碼器的輸入移除，以學習到的遮蔽向量取代。Review 節點、時間及與其他節點的關係保留。由 reviewer 的 User 節點取樣 3-hop，使用第一個 Review 可見的時間當切點。從 User 的 64 維向量預測被遮掉的 Review 原本是 `approved`、`commented` 或 `changes_requested`。以三類加權交叉熵用訓練期的 2,185 筆 Review 預訓練，11/1–11/8 有 532 筆驗證候選。

選擇驗證類別平均交叉熵最低的 checkpoint。以同一 checkpoint 和相同取樣種子，依序測完整遮蔽圖、將鄰居欄位向量歸零並保留邊和時間、移除訊息傳遞的邊。回到下游時，保留共用向量，凍結圖編碼器和 128→64 投影，在完整圖上使用原有切點及逐日 BCE 訓練 GRU。 [欄位遮蔽及消融實作](../scripts/relbench_review_mask.py)

結構檢查確認 2,185 筆訓練與 532 筆驗證樣本的目標 Review 都能從保留的 reviewer 邊取樣。將遮蔽位置的原始欄位改值不影響 logits，前向也不改動關聯邊。首批訓練的梯度 L2 在 PR 欄位 MLP、GraphSAGE 和 User 投影分別是 0.20694、0.92075、0.51960。

| 第 5 輪 checkpoint，532 筆驗證 Review | 三類平均交叉熵 | 三類 macro recall | Accuracy |
|---|---:|---:|---:|
| 只遮目標 Review 欄位 | **0.80210** | **67.04%** | **71.43%** |
| 另外將鄰居欄位向量歸零，保留邊與時間 | 0.99708 | 48.76% | 62.22% |
| 另外切斷所有訊息傳遞的邊 | 1.17573 | 33.33% | 5.83% |

預訓練在第 11 輪早停。三種驗證使用同一模型，切邊組固定猜 `changes_requested`。消融衡量這個 checkpoint 在當次重建中用了多少欄位與邊的訊號。

| 3-hop 預訓練方式 | 11/1–11/8 驗證總分 | 11/8–11/15 留出 macro-F1 | 留出 micro-F1 | 留出總分 |
|---|---:|---:|---:|---:|
| 原版隔天 BCE | 38.70% | 37.72% | 38.40% | **38.06%** |
| 遮蔽活動並猜類型 | 37.56% | 34.86% | 35.42% | 35.14% |
| 遮蔽 Review 欄位並保留連線 | 38.41% | 34.48% | 34.60% | 34.54% |

GRU 在 30 輪上限結束，選第 24 輪。有歷史組留出總分 35.17%，無歷史組 0%。已比對兩階段保存的 821 個圖編碼器與投影張量，確認凍結生效。比較採單一種子及留出週。表中的欄位遮蔽預訓練只重建 Review 的三種狀態，下游仍評估六類活動的發生。

```bash
HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1 \
  data/model/relbench_public/venv/bin/python scripts/relbench_asg2.py \
  --device cuda --hops 3 --daily-loss --review-feature-pretrain --pretrain-min-delta 0.001
```

分數、預訓練權重及最終 GRU 權重儲存在 `data/model/relbench_public/`，檔名以 `review_feature_gru_metrics_hop3` 開頭。

## Review 欄位遮蔽預訓練後的解凍比較

兩組從同一份第 5 輪遮蔽預訓練 checkpoint 載入圖編碼器，並以相同種子重新初始化 GRU。訓練資料、23 個切點、每批 512 筆、逐日 BCE、最多 30 輪及驗證早停規則一致。凍結組只訓練 GRU、預測頭、共用向量和每日位置向量。解凍組同時更新欄位編碼器、時間編碼器、GraphSAGE 與 128→64 投影，學習率為 `1e-4`。兩組的其他參數學習率都是 `1e-3`。

| 同一 Review 遮蔽 checkpoint | 最佳／實跑輪數 | 11/1–11/8 驗證總分 | 11/8–11/15 留出 macro-F1 | 留出 micro-F1 | 留出總分 |
|---|---:|---:|---:|---:|---:|
| 凍結圖編碼器 | 27／30 | 38.13% | 35.23% | 35.50% | 35.36% |
| 解凍圖編碼器與投影 | 2／8 | 38.95% | 36.00% | 36.79% | **36.39%** |

比對了兩組完整起始模型的 SHA-256，確認包含 GRU 在內的起始權重相同。凍結組最後的 821 個圖編碼及投影張量與來源權重完全一致，解凍組有 633 個張量實際改變。解凍組留出總分提高 1.03 個百分點。有歷史組從 36.06% 升至 37.15%，無歷史組皆為 0%。上一輪 34.54% 的凍結結果來自同一次執行完成預訓練再初始化 GRU，此處以相同初始化的凍結重跑分數作為配對基準。比較採單一種子及留出週。

```bash
export HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1
encoder=data/model/relbench_public/review_feature_gru_metrics_hop3.encoder.pt
python_relbench=data/model/relbench_public/venv/bin/python
"$python_relbench" scripts/relbench_asg2.py --device cuda --hops 3 --daily-loss \
  --encoder-checkpoint "$encoder"
"$python_relbench" scripts/relbench_asg2.py --device cuda --hops 3 --daily-loss \
  --encoder-checkpoint "$encoder" --encoder-finetune-lr 0.0001
```

成對結果分別寫入原預訓練檔名衍生的 `_frozen.json` 與 `_finetuned.json`。

## 固定切片與事件切點混合取樣

用同一套 3-hop RelBench，比較 24 小時和 12 小時固定切片、12 小時混入 20% 和 40% 事件切點，以及 6 小時混入 20%。五組每輪各用 36,000 筆從訓練期間抽出的使用者切點，訓練 3 輪原版六類 BCE。各組預測的未來長度與設定的 24、12 或 6 小時一致。

固定切片從台灣時間午夜開始。事件切點取事件與資料列都可見的時間再加一秒，觸發事件進入圖的歷史輸入。相鄰的事件切點要求同一人至少相隔 24 小時，取樣只用訓練期內有完整未來 24 小時可用的事件切點。用先前的逐日 GRU 訓練切點和週評分規則，凍結圖編碼器，固定取樣後快取使用者向量，重設預測頭到逐日訓練的先驗，GRU batch size 為 1024。在同一份驗證週選 epoch 與閾值，留出週納入 9,000 位使用者。 [混合取樣及快取](../scripts/relbench_mixed.py)

| 預訓練目標窗口 | 事件切點比例 | 含活動的樣本 | 11/1–11/8 驗證總分 | 11/8–11/15 留出總分 |
|---|---:|---:|---:|---:|
| 24 小時 | 0% | 6.62% | 38.70% | 36.21% |
| 12 小時 | 0% | 4.05% | 38.68% | 35.40% |
| 12 小時 | 20% | 11.64% | 37.81% | 36.21% |
| 12 小時 | 40% | 19.11% | 39.43% | 35.67% |
| 6 小時 | 20% | 8.98% | **39.76%** | **36.72%** |

在 12 小時目標內，20% 混合比固定切片高 0.81 個百分點。6 小時混入 20% 的留出 macro-F1 為 36.44%、micro-F1 為 36.99%，比等預算的 24 小時固定組高 0.51 個百分點。6 小時這輪只有混合組，結果涵蓋窗口長度及事件比例的共同影響。先前的 38.06% 使用全量 261,000 筆預訓練樣本，以及不同的 GRU 預測頭初值、訓練批次和圖取樣流程。五組比較採一個種子和一個留出週。

```bash
HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1 \
  data/model/relbench_public/venv/bin/python scripts/relbench_mixed.py --device cuda
```

分組權重及比較分數寫入 `data/model/relbench_public/mixed_sampling/`。

### 6 小時窗口提高事件比例

固定 6 小時預測窗口、每輪 36,000 筆樣本，以及同一套預訓練和 GRU 設定。以前次 20% 組為基準，加測 25%、30% 和 40%。事件切點共 14,976 個，各組從同一排序抽取前綴，固定時間樣本按各組剩餘預算從同一母體抽取。用原始規則重算 20% 組，確認總正例數與六類各自的正例數皆相同。

| 事件切點比例 | 含活動的樣本 | 11/1–11/8 驗證總分 | 11/8–11/15 留出總分 |
|---|---:|---:|---:|
| 20% | 8.98% | **39.76%** | 36.72% |
| 25% | 10.63% | 38.92% | **36.79%** |
| 30% | 12.24% | 38.87% | 36.61% |
| 40% | 15.53% | 39.20% | 36.48% |

25% 的留出分數只比 20% 高 0.08 個百分點。以驗證週選設定仍會選 20%。30% 與 40% 增加了有活動樣本的比例，兩組留出分數都較低。比較限於單一種子及留出週。

```bash
HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1 \
  data/model/relbench_public/venv/bin/python scripts/relbench_mixed.py --device cuda \
  --variant 6 25 --variant 6 30 --variant 6 40 \
  --output data/model/relbench_public/mixed_sampling_h6_more_events
```

三組的權重與分數寫入 `data/model/relbench_public/mixed_sampling_h6_more_events/`。

### 延長 6 小時預訓練並加入活動組合對比

沿用 6 小時監督目標、20% 事件切點及前次的 36,000 筆樣本。將預訓練從 3 輪增加到 12 輪，比較 BCE 與 `BCE + 0.01 × 對比 loss`。對比只使用六種標籤皆已知、有活動且有圖歷史的樣本。同一批中，不同使用者的六類標籤完全相同時形成正對，活動組合不同時形成負對，並按標籤重疊比例降低部分相似的負對權重。用 64 維向量的餘弦相似度與 0.2 溫度，讓向量靠得近的負對對 loss 影響較大。BCE 仍使用所有已知標籤。 [對比 loss 實作](../scripts/relbench_contrastive.py)

以同一批 36,000 筆驗證週固定切點計算第 3 和第 12 輪的短期 BCE。另在其中有活動的樣本計算六類 AP 的平均，檢查已知至少有一種活動時，模型能否分清類別。第 12 輪圖編碼器接同一套凍結、快取及 GRU 流程，沿用完整 cohort 週評分。

| 預訓練 | 六小時驗證 BCE | 活躍樣本 macro AP | 11/1–11/8 驗證總分 | 11/8–11/15 留出總分 |
|---|---:|---:|---:|---:|
| BCE 3 輪 | 0.012238 | 34.64% | **39.76%** | **36.72%** |
| BCE 12 輪 | 0.009647 | 33.77% | 37.37% | 33.93% |
| BCE＋對比 12 輪 | 0.009493 | 34.61% | 38.70% | 35.07% |

3 輪的短期驗證值取同一 BCE 長跑的第 3 輪，整週分數取前次相同取樣和 GRU 設定的 3 輪結果。兩個 12 輪版本的預訓練樣本指紋相同。對比在第 12 輪使用 2,784 個有正、負對的 anchor，當輪有 3,167 個有歷史和活動的候選樣本。相比同輪數 BCE，對比增加 0.84 個百分點的活躍樣本 macro AP、1.14 個百分點的週留出總分。短期診斷有 255 個活躍樣本，其中「開 issue＋留言且未開 PR」有 7 筆，「review＋開 PR」有 3 筆。訓練批中對應組合有 37 和 21 筆。結果限於這次取樣、種子及留出週。

```bash
export HF_HUB_CACHE="$PWD/data/model/relbench_public/hf_cache" HF_HUB_OFFLINE=1
python_relbench=data/model/relbench_public/venv/bin/python
"$python_relbench" scripts/relbench_mixed.py --device cuda --variant 6 20 --pretrain-epochs 12 \
  --encoder-validation-samples 36000 --output data/model/relbench_public/mixed_long_bce
"$python_relbench" scripts/relbench_mixed.py --device cuda --variant 6 20 --pretrain-epochs 12 \
  --encoder-validation-samples 36000 --contrastive-weight 0.01 \
  --output data/model/relbench_public/mixed_long_contrastive
```

各組分數與權重保存在 `data/model/relbench_public/mixed_long_bce/` 和 `mixed_long_contrastive/`。

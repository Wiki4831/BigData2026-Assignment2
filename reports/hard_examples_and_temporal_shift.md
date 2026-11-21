# ASG2 困難樣本、時間漂移與關聯模型

## 同一批使用者的逐週變動

以下使用相同的 9,000 位 cohort 成員。週起點為台灣時間週日 00:00。目標採原本的資料列可見規則，六個標籤週均有完整的小時檔覆蓋。

| 週起點 | 至少一種活動 | 開 PR | 開 issue | Issue 留言 | Review | Review 留言 | Push |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10/4 | 2,437 | 473 | 249 | 319 | 313 | 154 | 2,184 |
| 10/11 | 2,375 | 393 | 236 | 233 | 280 | 187 | 2,130 |
| 10/18 | 2,070 | 348 | 225 | 212 | 225 | 171 | 1,848 |
| 10/25 | 1,413 | 333 | 177 | 224 | 239 | 75 | 1,009 |
| 11/1 | 952 | 383 | 196 | 256 | 275 | 105 | 283 |
| 11/8 | 868 | 338 | 182 | 205 | 233 | 69 | 277 |

10/4 週到 11/8 週，觀察到 push 的比例從 24.27% 降到 3.08%。固定只看 10/4 切點前就有可用圖歷史的 1,934 人，比例仍從 66.44% 降到 10.19%。在有開 PR 的人中，同週觀察到 push 的比例從 85.84% 降到 27.81%。「上週有 push，這週也有 push」在 10/11 週是 62.04%，在 11/1 週是 14.97%，在 11/8 週是 26.15%。這代表標籤比例與可觀察到的活動組合都有明顯變化。 [逐週統計及標籤組合](../data/analysis/hard_examples_temporal_shift.json)

重查四個 GH Archive 原始小時。台灣時間 10/14 12:00 的檔案有 86,415 筆全站事件，其中 68,184 筆為 `PushEvent`。10/15 04:00、10/16 12:00、11/10 20:00 的抽查檔仍各有 4,866、6,196、3,779 筆其他事件，`PushEvent` 都是 0 筆。這提供了來源收錄變動的直接線索。抽查尺度為小時，能支持觀測變了的判斷。 [既有來源抽查](../notebooks/02_user_activity_timing.ipynb)

「照抄上週」的整週分數也從 10/11 週 45.62% 降到 11/8 週 32.85%。這個固定規則能用來檢查切分難度。模型的時間切分、活動分布和架構效果需要分開量。 [對照基準](time_encoder_ablation.md#參考週差異)

## 目前模型認錯的難例

使用留出週的 3-hop 儲存推論、原本驗證閾值，固定在有可用圖歷史的使用者。活動類型為 `open_pr` 時，實際有其他活動但未開 PR 的 494 人裡，有 127 人被誤報，誤報率是 **25.71%**。六類活動都為 0 的 3,303 人中有 177 人被誤報，為 **5.36%**。以同一批有歷史的開 PR 正例計算分數排序 AUC，對「有其他活動」負例是 0.636，對全 0 負例是 0.828。

使用者舉的具體組合也有重疊。開 issue 且留言、未開 PR 的 32 人中有 16 人被判為開 PR。Review 且開 PR 的 49 人中有 31 人被判為開 PR。兩組直接比較的 open-PR AUC 為 0.575。這兩組的規模小，適合作為案例。主要衡量方式應涵蓋各類所有仍有活動的負例。 [逐人推論 CSV](../data/model/relbench_public/hop3_holdout_predictions.csv)

## 困難樣本訓練候選

| 方法 | 放進這個任務的方式 | 能量到的效果 |
|---|---|---|
| 逐類困難負例與排序輔助 loss | 對 `open_pr` 抽實際開 PR 的正例，從同時段、有其他活動的已知負例中找模型 PR 分數較高者。盡量配對相近的歷史活躍度或已知 repo 背景。保留完整 BCE，再加 `softplus(負例 logit − 正例 logit)`。 | 直接處理 PR 分數在有其他活動時的誤報。BCE 繼續學安靜時段與多標籤分布。 [RankNet 原文 §2](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf) |
| Asymmetric Loss | 對六類各自降低容易的負例權重，保留正例和高分錯誤，沿用相同樣本與驗證。 | 提供不需配對的多標籤對照。資料收錄可疑的活動需要按可觀測性處理。 [Asymmetric Loss 原文](https://arxiv.org/abs/2009.14119) |
| 標籤感知的對比學習 | 以類型或標籤重疊定義配對強度，讓共享 review 標籤的樣本仍保有部分相似性，同時對不同的 `open_pr` 狀態施加區分。 | 避免單一向量被要求把所有不同活動組合等量推開，也減少批內等候完整六位元標籤重複的依賴。 [MulSupCon 原文](https://ojs.aaai.org/index.php/AAAI/article/view/29619) |

先前的實驗以六類標籤完全相同作對比正例。相同 12 輪下，整週留出總分從 33.93% 回到 35.07%，但原本 3 輪的 36.72% 仍較高。在 36,000 筆六小時訓練樣本中，兩種具體組合分別只有 37 與 21 筆。逐類配對能把這些例子用在 `open_pr` 的正負關係上，完整組合仍可作分組診斷。挑選樣本的模型分數與超參數應只用訓練和當折的驗證資料建立。 [原有對比對照](time_encoder_ablation.md#延長-6-小時預訓練並加入活動組合對比)

## 換 RelGNN 或關聯式 Transformer

- **RelGNN**。以資料表的 FK 自動找原子路徑，將多外鍵列當作橋或 hub。ASG2 的 `PR → PRReviews → reviewer` 與 `repo → Pushes → user` 都符合經同一中介列連到兩個外鍵端點的形式。原文以複合注意力處理這些路徑，提供直接測試聚合方式的對照。可保留時間取樣、User 向量及 GRU。這是對本 schema 的應用推論。 [RelGNN 原文](https://arxiv.org/abs/2502.06784) · [官方實作](https://github.com/snap-stanford/RelGNN)
- **Relational Transformer**。原文把 cell 當 token，以同欄、同列及外鍵鄰居的 attention mask 取用不同關係，另有全域注意力。用沿 FK 取得的有界上下文做遮蔽 cell 預測。官方 repo 提供 RelBench 格式的自有資料庫轉接及較新的 RT-J checkpoint。可把每個使用者切點的六個布林目標放進任務表做獨立對照，這是本任務的轉接設計。 [Relational Transformer 原文](https://arxiv.org/abs/2510.06377) · [官方自有資料庫範例](https://github.com/stanford-star/relational-transformer)

先用現有 3-hop 編碼器在 11/1、11/8 兩個滾動留出週測近期加權及逐類難例，所有訓練窗口在該折驗證前結束。控制抽樣預算和 optimizer 更新數，記錄完整六類 Score、各類 F1 與活躍負例誤報率。可另報五類診斷分數以查看 push 變動對比較的影響。再固定已選的損失和資料配方，先換 RelGNN，再視結果比較 Transformer。樣本、目標或架構每次只改一項，便於解讀差異。

## 論文庫核對位置

- **RelGNN: Composite Message Passing for Relational Deep Learning**，ID `46f46f0699d1224d`，`CNN for Trajectory/E - RelGNN- Composite Message Passing for Relational Deep Learning.pdf`，本次核對 PDF 第 3–6 頁，原子路徑定義在第 5 頁，注意力在第 6 頁。
- **Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data**，ID `1a1e18d5842b76c3`，`CNN for Trajectory/E - Relational Transformer- Toward Zero-Shot Foundation Models for Relational Data.pdf`，本次核對 PDF 第 3–6 頁，cell 表示與取樣在第 4 頁，注意力在第 5 頁，遮蔽 loss 與時間切分在第 6 頁。

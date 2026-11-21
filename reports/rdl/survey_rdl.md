# RDL 文獻導讀：5 篇，建立基礎概念

目的：了解 Relational Deep Learning（RDL）是什麼、怎麼演進、和我們的作業有什麼關係。順序是「先懂想法，再看怎麼改進」。

證據等級：標【庫】的三篇來自本機論文庫的 L1 概覽筆記，筆記未逐頁核對原文；標【摘要】的兩篇只讀了 arXiv 摘要頁（論文庫沒有這兩篇），細節沒有核對。引用數字都是作者自己報告的。

## 閱讀順序與內容

| 順序 | 論文 | 一句話 | 來源 |
|---|---|---|---|
| 1 | Fey 等人（2023，ICML 2024），*Relational Deep Learning: Graph Representation Learning on Relational Databases*，arXiv:2312.04615 | 提出 RDL 的想法：把資料庫變成圖，每一列是節點、外鍵是邊，用 GNN 直接學，不要人工合併資料表 | 【摘要】[arXiv](https://arxiv.org/abs/2312.04615) |
| 2 | Robinson 等人（2024，NeurIPS 2024），*RelBench: A Benchmark for Deep Learning on Relational Databases*，arXiv:2407.20060 | 提供 7 個資料庫、30 個預測任務和 Python 套件；作者報告 RDL 比資深資料科學家人工做特徵的模型好，且人力少了一個數量級以上 | 【摘要】[arXiv](https://arxiv.org/abs/2407.20060) |
| 3 | Chen、Kanatsoulis、Leskovec（ICML 2025），*RelGNN: Composite Message Passing for Relational Deep Learning*，arXiv:2502.06784 | 改進訊息傳遞：把「來源→中介表→目的」當一條原子路徑，一次聚合，避免經過中介節點反覆混入無關訊息 | 【庫】論文 ID 46f46f0699d1224d |
| 4 | Dwivedi 等人（ICLR 2026），*Relational Graph Transformer*，arXiv:2505.10960 | 用 Transformer 取代逐層訊息傳遞：每個節點變成帶有特徵、類型、hop 距離、時間、局部結構的 token，用局部注意力加全域中心向量 | 【庫】論文 ID f40b6bc8fab160ce |
| 5 | Ranjan 等人（ICLR 2026），*Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data*，arXiv:2510.06377 | 把每個儲存格當 token，沿外鍵做注意力，預訓練後可以直接用在沒看過的資料庫（作者報告零樣本達到全監督的 90.3%） | 【庫】論文 ID 1a1e18d5842b76c3 |

補充（沒列入 5 篇）：ContextGNN（Yuan 等人，ICLR 2025，arXiv:2411.19513）處理 RDL 的推薦任務，用「使用者互動過的物品」和「新物品」兩種分支再融合。與我們的分類任務關係較遠。[arXiv](https://arxiv.org/abs/2411.19513)

## 基礎概念（讀完前兩篇該懂的事）

| 概念 | 白話 |
|---|---|
| 為什麼要 RDL | 傳統做法要先把多張表合併成一張大表（人工寫「近 30 天下單次數」之類的特徵）。這很花時間，也可能丟掉關係訊號。RDL 把這件事交給模型 |
| 圖怎麼建 | 一列 = 一個節點，主鍵與外鍵 = 邊。節點有類型（來自哪張表） |
| 預測任務長什麼樣 | 一張任務表：每列是「哪個實體、在哪個時間點、標籤是什麼」。我們的任務就是「使用者 × 切點 → 六個 0/1」 |
| 時間 | 預測時間點之前的列才能用；所以要「時間鄰居取樣」，只從切點前已出現的列取鄰居 |
| 欄位怎麼進模型 | 每個欄位（數字、類別、時間、文字）先由欄位編碼器變成向量，再丟進 GNN |
| 基線 | RelBench 用 GraphSAGE（把鄰居向量加總）當標準 RDL 模型；我們用的就是它 |

## 三條改進路線（第 3–5 篇）

1. **改訊息怎麼傳（RelGNN）。** 針對「中介表」，例如我們的 Push 同時連使用者和 repo，使用者要經過它才能看到 repo。RelGNN 把這種 2 條邊合成 1 步。我們試過，反而低約 3 點，推測因為我們的中介節點（push、review）本身有時間和屬性，合併後資訊被壓縮。
2. **改用 Transformer（RelGT）。** 不逐層平均鄰居，而是讓取樣到的鄰居 token 彼此做注意力，並明確告訴模型每個 token 的類型、離中心幾 hop、發生在多久以前。對應我們目前 GraphSAGE 的弱點：分不出「一天前的 push」和「一個月前的 push」，所以我們另外加了時間編碼和使用者統計特徵。
3. **預訓練與基礎模型（Relational Transformer）。** 用遮罩預測在很多資料庫上預訓練。對應我們的自監督預訓練（次日預測），想法一致：標籤少時，用大量不需標註的任務先學表示。

## 對我們作業的啟發

- 我們的核心就是 RDL（第 1、2 篇的設定），做 Q2、Q3 的說明時可以直接引用。
- RelGNN（第 3 篇）是我們實作過的變體，結果是負面的，可以誠實寫在報告裡。
- RelGT（第 4 篇）是目前最值得嘗試、但還沒做的一條路線：它把「類型、hop、時間」顯式變成 token 輸入，對應 Q3 的時間建模。實作成本中等（需要接論文的程式）。
- 第 5 篇說明預訓練的價值，可以當自監督預訓練的依據。

## 未核對事項

- 第 1、2 篇只讀了摘要，圖的建構細節、時間鄰居取樣和任務表的定義，要在寫報告前核對原文。
- 第 3–5 篇的實驗數字是筆記轉述，未核對 PDF 頁碼。

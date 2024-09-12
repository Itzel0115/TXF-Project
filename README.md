# TXF Quant Research

此為參與 2024 暑期 tmba程式交易部門期間，我所實作的一些cta策略專案統整。

本專案為**台指期（TXF）1 分鐘日盤**資料的量化研究與回測框架，涵蓋技術指標策略、網格搜尋、風險覆蓋、Walk-Forward 與 Meta-Labeling。

- **資料**：1 分鐘 K 線、欄位 `datetime, Open, High, Low, Close, Volume`
- **策略**：均線交叉（MA）、布林均值回歸（BB）、海龜突破（TURTLE）、ML 方向預測、Meta-Labeling
- **輸出**：所有績效表統一寫入 `outputs/` 目錄的 CSV，方便統整與版本管理

---

## 安裝與執行

### 1. 安裝環境

```bash
pip install -r requirements.txt
```

### 2. 準備資料

將 1 分鐘日盤 CSV 放在**專案根目錄**，檔名：`TXF_R1_1min_data_combined.csv`。  
欄位需包含：**datetime**（或 date / time / timestamp）與 **Open, High, Low, Close, Volume**（大小寫不拘）。

### 3. 執行策略優化與評估

在專案根目錄執行：

```bash
python run_optimization.py
```

程式會依序：

1. **網格搜尋**：MA / BB / TURTLE × 多組參數 × 多時間週期（5/10/15/30/60 分鐘）× 日盤/全時段
2. **各類型最佳**：每種策略取 Sharpe 最佳一組，寫入 `outputs/strategy_best_by_type.csv`
3. **風險覆蓋**：對上述最佳候選加上波動目標與回撤降槓桿，寫入 `outputs/strategy_best_with_risk_overlay.csv`
4. **Walk-Forward**：對整體最佳候選做滾動樣本外評估，寫入 `outputs/walk_forward_best_overall.csv`
5. **Meta-Labeling**：以 MA 最佳為基底訊號、ML 篩選進出場，寫入 `outputs/meta_labeling_comparison.csv`

**結果存放位置**：所有 CSV 皆在 **`outputs/`** 目錄下。

---

## 輸出檔案說明（outputs/）

| 檔案名稱 | 內容說明 |
|----------|----------|
| `strategy_grid_search_results.csv` | 網格搜尋全部組合的績效（年化報酬、波動、Sharpe、最大回撤、交易次數等） |
| `strategy_best_by_type.csv` | 每種策略類型（MA / BB / TURTLE）中 Sharpe 最佳的一筆 |
| `strategy_best_with_risk_overlay.csv` | 上述最佳候選加上風險覆蓋後的績效 |
| `walk_forward_best_overall.csv` | 整體最佳候選的 Walk-Forward 各 fold 與合併樣本外績效 |
| `meta_labeling_comparison.csv` | Meta-Labeling 之基底策略 vs 加上 ML 篩選後的績效與準確率等 |

---

## 專案結構

```
TXF/
├── README.md
├── requirements.txt
├── TXF_R1_1min_data_combined.csv   # 原始資料
├── run_optimization.py             # 主程式：網格搜尋 + 風險覆蓋 + WF + Meta-Labeling
├── outputs/                        # 所有策略績效 CSV 輸出
│   ├── strategy_grid_search_results.csv
│   ├── strategy_best_by_type.csv
│   ├── strategy_best_with_risk_overlay.csv
│   ├── walk_forward_best_overall.csv
│   └── meta_labeling_comparison.csv
└── src/
    ├── config/                     # 設定範例（YAML）
    ├── data/                       # 載入、清理、重採樣
    ├── features/                  # 技術指標
    ├── strategies/                # MA、布林、海龜
    ├── backtest/                   # 回測引擎、風險覆蓋
    ├── ml/                         # 方向預測、Meta-Label 資料集與模型
    ├── evaluation/                 # 績效彙總、報表
    ├── visualization/              # 淨值、回撤、月度 heatmap
    └── utils/                      # 時間、報酬率工具
```

---

## 如何解讀結果

- **strategy_grid_search_results.csv**：可依 `sharpe_ratio`、`annualized_return` 排序，挑出表現較佳的參數與週期。
- **strategy_best_by_type.csv**：快速比較「MA vs BB vs TURTLE」在當前資料上的最佳代表。
- **strategy_best_with_risk_overlay.csv**：檢視加上波動目標與回撤控管後的報酬/回撤/交易次數變化。
- **walk_forward_best_overall.csv**：檢視樣本外是否穩定；`fold` 為各區間，`ALL_TEST` 為整段樣本外彙總。
- **meta_labeling_comparison.csv**：比較 `base_test`（僅基底訊號）與 `meta_test`（ML 篩選後）的績效與 `meta_accuracy` / `meta_f1` / `meta_auc`。

以上即為本專案之統整：**單一入口 `run_optimization.py`、所有策略結果集中於 `outputs/` 的 CSV**。

---

## 本次實際執行結果摘要（來自 `outputs/`）

以下數字皆由目前 `outputs/` 目錄下的 CSV 檔計算而來，可視為本專案在 `TXF_R1_1min_data_combined.csv` 這份資料上的「代表性實驗結果」。

### 1. 各類型最佳策略（`strategy_best_by_type.csv`）

- **MA 類（均線交叉）最佳：`MA_day_15min_50_200_L`**
  - 資料：只用日盤、重採樣成 15 分鐘
  - 年化報酬：約 **8.1%**
  - 年化波動：約 **11.2%**
  - Sharpe：約 **0.73**
  - 最大回撤：約 **-21.6%**
  - 交易次數：**503 筆**  
  ⇒ 在三類策略中，MA 類在此資料上的風險調整後報酬（Sharpe）最佳。

- **TURTLE 類（海龜突破）最佳：`TURTLE_day_30min_80_30_20`**
  - 30 分鐘、日盤
  - 年化報酬：約 **3.7%**、Sharpe 約 **0.27**、最大回撤約 **-27.5%**、交易約 **902 筆**  
  ⇒ 有正報酬，但 Sharpe 明顯低於 MA。

- **BB 類（布林均值回歸）最佳：`BB_day_60min_100_3.0`**
  - 60 分鐘、日盤
  - 年化報酬約 **0.17%**、Sharpe 約 **0.05**、最大回撤約 **-14.5%**  
  ⇒ 在這份資料上布林類表現偏弱，幾乎持平。

整體來看，在目前設定與樣本下，**長短期均線交叉（50/200, 15min, 日盤）是最有代表性的 CTA 策略**。

### 2. 加入風險覆蓋後的表現（`strategy_best_with_risk_overlay.csv`）

對上述三個「最佳」策略套用統一的風險覆蓋（波動目標 / 回撤降槓桿），觀察變化：

- **MA_day_15min_50_200_L（加風險覆蓋）**
  - 年化報酬：約 **7.4%**（略低於原始 8.1%）
  - 年化波動：約 **12.7%**
  - Sharpe：約 **0.58**
  - 最大回撤：約 **-21.5%**
  - 交易次數：**49,668 筆**（因槓桿調整頻繁，使得「部位變化」次數大幅增加）  
  ⇒ 報酬略降、風險略升，但 Sharpe 仍維持中等水準；風險覆蓋在此設定下更像是「槓桿調整 + 風險控制骨架」，實務上可再調整參數。

- **TURTLE 與 BB** 在加上風險覆蓋後，Sharpe 有小幅提升（Turtle 約 0.31），但整體仍明顯落後 MA 類。

結論是：**MA 策略在加上統一的風險覆蓋後仍是三者之中最穩定的主力**。

### 3. Walk-Forward 樣本外表現（`walk_forward_best_overall.csv`）

對整體最佳候選（實務上是 `MA_day_15min_50_200_L` 加風險覆蓋）做多段樣本外 Walk-Forward，重點如下：

- 各 fold 年化報酬大多在 **7% ~ 18%** 區間，Sharpe 介於 **0.55 ~ 1.35** 之間（其中第 4 段明顯偏弱，Sharpe 約 0.09）
- 合併所有樣本外區間（`ALL_TEST`）：
  - 年化報酬：約 **9.5%**
  - 年化波動：約 **13.2%**
  - Sharpe：約 **0.72**
  - 最大回撤：約 **-20.6%**

這代表：**在不同時間區間與市場環境中，此 MA 策略的整體表現仍維持中度正向的風險調整後報酬，且最大回撤約落在 -20% 左右，是實務上可以接受、但仍需搭配資金管理的風險水準**。

### 4. Meta-Labeling 結果（`meta_labeling_comparison.csv`）

以 MA 最佳策略作為「基底訊號」，再用 ML 做 Meta-Labeling（判斷哪些訊號要執行）後：

- **base_test（僅基底策略）**
  - 年化報酬：約 **11.8%**
  - 年化波動：約 **15.6%**
  - Sharpe：約 **0.75**
  - 最大回撤：約 **-21.6%**

- **meta_test（加上 Meta-Labeling）**
  - 年化報酬：約 **-1.2%**（轉為小幅虧損）
  - 年化波動：約 **13.7%**
  - Sharpe：約 **-0.09**
  - 最大回撤：約 **-17.5%**
  - Meta 模型表現：`meta_accuracy ≈ 0.495`、`meta_f1 ≈ 0.405`、`meta_auc ≈ 0.504`（接近亂猜）

這部分顯示：**在目前的特徵、標籤與模型設計下，Meta-Labeling 並沒有為基底策略帶來正面貢獻，甚至顯著拖累績效**。  
實務上若要讓 Meta-Labeling 有效，可能需要：

- 更精細的標籤設計（例如風險報酬比、持倉期間限制）
- 更貼近微結構的特徵（委託簿、不平衡度等）
- 嚴謹的樣本外驗證與正則化，避免過度擬合。


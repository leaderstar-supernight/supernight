# WHID Windows 移機與 Codex 交接說明

更新日期：2026-09-05
正式執行環境：Windows
專案根目錄：本文件所在的 `Stock_Valuation` 資料夾

## 1. 先給新電腦 Codex 的結論

這是一套分成兩階段、涵蓋台股與美股的股票研究系統。完整每日流程由：

```text
Stock_price_prediction/daily_pipeline.py
```

統一依序執行：

1. `Stock_Valuation_ReDesgin_TW.ipynb`：台股第一階段，基本面與估值。
2. `Stock_price_prediction/TW_Timing_第二階段.ipynb`：台股第二階段，趨勢、籌碼、風險與 AI 研究。
3. `Stock_Valuation_ReDesgin_US.ipynb`：美股第一階段，基本面與估值。
4. `Stock_price_prediction/US_Timing_第二階段.ipynb`：美股第二階段，趨勢、量價、風險與 AI 研究。
5. 封存當日預測，並結算已滿 10 個交易日的舊預測。

程式主要使用相對路徑。只要完整保留本資料夾的內部結構，搬到新磁碟或不同 Windows 使用者目錄後，通常不必修改程式碼中的路徑。

`Stock_Valuation_SMART.ipynb` 不是目前每日正式流程的一部分。請保留作歷史參考，但不要加入排程。

## 2. 目前系統設計

### 第一階段：WHID 基本面與估值

- 台股以 FinMind 為主要資料來源。
- 台股只有 FinMind 缺少的預估 EPS，以及必要時缺少的 EBITDA、財務安全性資料與 ROIC，才由 Yahoo 補充。
- 美股以 Yahoo 為主要資料來源。
- 產出詳細版與簡化版 Excel，兩者分開保存。
- 基本面、買價、盈餘品質、護城河、財務安全性與獲利品質等評價保留各自依據。

### 第二階段：時機、籌碼、風險與 AI 研究

- 接收第一階段候選股票，再研究進出場時機。
- 趨勢分成短期、中期、長期；目前只有中期趨勢參與既有進出場規則，短期與長期先作報表參考。
- 台股包含法人、融資融券、借券及可取得時的籌碼 K 線／券商分點資料。
- 美股使用 Yahoo 的價量與市場資料；沒有台灣市場的三大法人、融資融券制度資料。
- AI 研究目標是未來 10 個交易日「上行先觸／下行先觸／盤整」三分類機率。
- AI 結果目前不直接決定「未持有建議」。歷史樣本不足也不代表整份技術與籌碼研究失效。

詳細定義請閱讀：

- `Stock_price_prediction/00_從這裡開始.md`
- `Stock_price_prediction/README_第二階段.md`
- `Stock_price_prediction/README_美股第二階段.md`
- `Stock_price_prediction/WHID第二階段_指標定義與判讀速查.docx`
- `Stock_price_prediction/每日驗證與30日評估.md`

## 3. 搬移時必須完整保留的內容

建議直接複製整個 `Stock_Valuation` 資料夾。以下內容尤其不能遺漏：

| 內容 | 位置 | 原因 |
| --- | --- | --- |
| 四份正式 Notebook | 專案根目錄及 `Stock_price_prediction` | 每日流程主體 |
| 第二階段 Python 模組 | `Stock_price_prediction/*.py` | Notebook 實際呼叫的分析、AI 與報表邏輯 |
| 第一階段真實設定 | `config/TW.yaml`、`config/US.yaml`、`config/Fin.yaml` | 股票名單、權重、FinMind token 等 |
| 第二階段設定 | `Stock_price_prediction/config/timing_TW.yaml`、`timing_US.yaml` | 候選名單、門檻、輸出與回測設定 |
| 歷史報表 | 專案根目錄 `reports` | 日後查閱與候選名單來源 |
| 驗證資料庫 | `system_data/validation/prediction_audit.sqlite3` | 累積預測、10 日結算及 30 日評估歷史 |
| 說明文件 | `Stock_price_prediction/*.md`、`.docx` | 指標定義與操作說明 |

`config/TW.yaml` 和 `config/Fin.yaml` 含 FinMind token。只可經可信任的私人裝置、加密磁碟或受控傳輸方式搬移，不要上傳公開 GitHub，也不要把 token 貼進 Codex 對話。

以下內容即使不複製也能重建；完整搬移時保留也沒有問題：

- `__pycache__`
- `.ipynb_checkpoints`
- `system_data/cache/TW`
- `system_data/cache/US`
- `system_data/docx_render_check`

請勿只靠 GitHub clone 完成正式移機。版本庫通常不包含真實 YAML、每日日報、快取及驗證 SQLite，因此會遺失 FinMind 設定與已累積的預測歷史。

## 4. 複製前的安全步驟

1. 等待當日 `daily_pipeline.py` 完全結束。
2. 關閉 Jupyter、Python、Excel，以及正在讀取 WHID 報表的程式。
3. 確認沒有程式正在寫入 `prediction_audit.sqlite3`。
4. 再複製整個 `Stock_Valuation` 資料夾。
5. 新電腦驗證成功前，舊電腦排程可先保留但暫停執行。
6. 新電腦正式啟用後，務必停用舊電腦的每日排程，避免兩台電腦同時產生報表或各自寫入不同版本的驗證台帳。

若複製時仍有流程執行中，SQLite 與當天報表可能只複製到一半。最安全的作法是流程結束、相關程式關閉後再複製。

## 5. 新 Windows 電腦的 Python 環境

目前驗證過的基準環境是：

- Python 3.12.8
- numpy 1.26.4
- pandas 2.2.2
- PyYAML 6.0.1
- requests 2.32.3
- yfinance 1.7.0
- openpyxl 3.1.5
- scikit-learn 1.5.1
- xgboost 3.4.1
- torch 2.13.0
- nbformat 5.10.4
- nbclient 0.8.0
- scipy 1.13.1
- ipykernel 6.28.0

建議在新電腦建立獨立 Conda 環境，不要複製舊電腦的 Anaconda 環境資料夾或 `.venv`。範例：

```powershell
conda create -n whid python=3.12 -y
conda activate whid
python -m pip install -r .\Stock_price_prediction\requirements-timing.txt
python -m pip install -r .\Stock_price_prediction\requirements-timing-US.txt
python -m pip install ipykernel
python -m ipykernel install --user --name pytorch-gpu --display-name "Python (PyTorch GPU)"
```

以上命令應在新電腦複製後的 `Stock_Valuation` 根目錄執行。

三份正式 Notebook 的 metadata 使用 `pytorch-gpu` 核心；台股第二階段使用 `python3`。因此新電腦至少要：

1. 在 WHID 環境安裝 `ipykernel`。
2. 註冊名稱為 `pytorch-gpu` 的核心。
3. 確認 `python3` 核心也能使用同一套必要套件。

若新電腦沒有 NVIDIA GPU，核心名稱仍可保留為 `pytorch-gpu`。PyTorch 使用 CPU 也能執行，只是 LSTM 訓練較慢。FinLab 屬於台股額外功能；若要啟用需要 FinLab 的項目，再依其授權與官方安裝方式處理。沒有 FinLab 時，程式應使用既有降級邏輯，不能因此中斷整條每日流程。

## 6. 新電腦必須檢查的設定

### 第一階段

檢查以下檔案存在且可讀取：

```text
config/TW.yaml
config/US.yaml
config/Fin.yaml
```

只檢查 token 是否已設定，不要在終端、截圖、日誌或對話中顯示 token 本身。

### 第二階段

檢查：

```text
Stock_price_prediction/config/timing_TW.yaml
Stock_price_prediction/config/timing_US.yaml
```

目前設定使用相對位置：

- 台股 FinMind 設定指向專案根目錄的 `config/TW.yaml`。
- TW／US 兩階段 Excel 都指向專案根目錄 `reports/詳細版` 與 `reports/簡化版`。
- 第二階段研究、快取及驗證資料指向專案根目錄 `system_data`。

只要資料夾層級沒有被拆開，移機後不應把這些值改成某位 Windows 使用者的絕對路徑。

Notebook 可能保留舊執行輸出。畫面中看到舊電腦的 `C:\Users\...` 僅是歷史輸出文字，不代表目前程式仍綁定舊路徑。判斷是否需要修改時，應檢查程式 cell，而不是已執行輸出。

## 7. 第一次驗證順序

在新電腦開啟 PowerShell，進入複製後的 `Stock_Valuation` 根目錄，並啟用 WHID 環境。

### 第一步：舊版資料夾搬移（舊專案升級時執行一次）

```powershell
python .\migrate_report_layout.py
```

這個工具可重複執行，會重新命名舊報表、搬移研究與驗證資料，並更新 SQLite 內保存的舊報表路徑。全新 clone 且沒有舊報表時可略過。

### 第二步：路徑與結構測試

```powershell
python .\Stock_price_prediction\daily_pipeline.py --dry-run
```

預期結果：

- `status` 為 `dry_run`。
- 能找到 TW、US 各兩份 Notebook。
- 能讀取既有驗證資料庫狀態。

### 第三步：確認驗證歷史仍在

```powershell
python .\Stock_price_prediction\prediction_audit.py status
```

若數字全部從零開始，先不要執行正式流程；應確認 `prediction_audit.sqlite3` 是否漏複製或放錯資料夾。

### 第四步：執行一次完整流程

```powershell
python .\Stock_price_prediction\daily_pipeline.py
```

完整流程會連接 FinMind 與 Yahoo，執行時間依股票數、網路及 AI 訓練狀態而異。完成後找到最新的：

```text
system_data/validation/每日執行紀錄/YYYY/MM/DD/執行編號/summary.json
```

四個 Notebook 都應為 `status: ok`，最外層 `status` 也應為 `ok`。單一股票在結算時出現「原始價或還原價不可用」可列為資料警告；只要四階段與最外層狀態為 `ok`，就不代表整條流程失敗。

### 已知無害警告

Windows 執行 Notebook 時可能顯示：

- `Proactor event loop does not implement add_reader...`
- `Debugger warning: frozen modules are being used...`

這兩類訊息來自 Jupyter、ZMQ 或 debugger，通常不影響報表。應以 `summary.json`、Excel 是否產出，以及排程的返回碼是否為 0 判斷是否成功。

## 8. 報表與驗證資料位置

| 用途 | 位置 |
| --- | --- |
| 台股詳細版（Step1、Step2） | `reports/詳細版/TW` |
| 台股簡化版（Step1、Step2） | `reports/簡化版/TW` |
| 美股詳細版（Step1、Step2） | `reports/詳細版/US` |
| 美股簡化版（Step1、Step2） | `reports/簡化版/US` |
| 第二階段研究資料 | `system_data/research/TW`、`system_data/research/US` |
| 每次自動執行紀錄 | `system_data/validation/每日執行紀錄` |
| 預測驗證資料庫 | `system_data/validation/prediction_audit.sqlite3` |
| 30 日評估 | `reports/30日評估/YYYY/MM/WHID_30日驗證_YYYYMMDD_HHMMSS.xlsx` |

舊版短線 AI 報表保存在 `system_data/legacy_reports/short_term_ai`，不是目前新版第二階段的正式輸出。

## 9. Windows 每晚 20:00 排程

排程不會隨專案資料夾自動搬到新電腦，必須在新 Windows 重新建立。

建議設定：

| 項目 | 值 |
| --- | --- |
| 名稱 | `WHID Daily Stock Evaluation` |
| 時間 | 每天台北時間 20:00 |
| 程式 | 新電腦 WHID Conda 環境的 `python.exe` |
| 參數 | 新電腦 `Stock_price_prediction/daily_pipeline.py` 的完整路徑 |
| 起始位置 | 新電腦的 `Stock_price_prediction` 資料夾 |
| 錯過時間後執行 | 開啟 |
| 喚醒電腦執行 | 開啟 |
| 重複執行 | 不允許新執行個體與舊執行個體重疊 |
| 失敗重試 | 每 15 分鐘一次，共 2 次 |
| 最長執行 | 6 小時 |
| 登入方式 | 使用者已登入時執行 |

「使用者已登入時執行」代表螢幕鎖定仍可執行，但使用者必須保持 Windows 登入狀態。Codex App 不需要保持開啟。電腦若關機，必須依賴「錯過時間後儘快執行」；電腦處於無法喚醒的關機狀態時仍不會準時啟動。

建立後先手動觸發一次，並檢查：

1. 上次執行結果為 `0`。
2. 最新 `summary.json` 最外層為 `ok`。
3. 四份新 Excel 都有產出。
4. 驗證資料庫的統計數字有合理增加。

Codex App 內原有的自動化也不會隨資料夾複製。若需要 20:30 的執行結果檢查或每月預測準確性評估，應在新電腦驗證正式 Windows 排程後，再由新電腦 Codex 重新建立。Windows 工作排程器是正式執行來源，Codex 自動化只作檢查或提醒，避免兩邊都啟動同一條 pipeline。

## 10. 月度／30 日評估

手動檢查及產出評估報告：

```powershell
python .\Stock_price_prediction\prediction_audit.py status
python .\Stock_price_prediction\prediction_audit.py review
```

AI 預測要經過未來 10 個交易日後才會成熟；基本面與估值則需要更長觀察期。詳細門檻以 `Stock_price_prediction/每日驗證與30日評估.md` 為準，不要因移機後短期樣本不足就改變模型權重。

## 11. 新舊電腦切換檢查表

- [ ] 已完整複製 `Stock_Valuation`。
- [ ] 三份真實 YAML 與兩份 timing YAML 均存在。
- [ ] `prediction_audit.sqlite3` 已保留，歷史筆數不是零。
- [ ] 已在新電腦重建 Python 環境，沒有複製舊電腦環境目錄。
- [ ] `pytorch-gpu` 與 `python3` Jupyter 核心可用。
- [ ] `daily_pipeline.py --dry-run` 成功。
- [ ] 完整流程四個階段均成功。
- [ ] 第一、第二階段 TW／US 報表均有新檔。
- [ ] 新電腦 Windows 20:00 排程已建立並測試。
- [ ] 新電腦驗證成功後，舊電腦正式排程已停用。
- [ ] 同一時間只有一台電腦寫入正式 reports 與驗證資料庫。

## 12. 可直接貼給新電腦 Codex 的接手指令

```text
請先閱讀專案根目錄的 WINDOWS_移機交接說明.md，以及 Stock_price_prediction/00_從這裡開始.md 和 每日驗證與30日評估.md。

這是從舊 Windows 完整複製來的 WHID 股票研究專案。請先做唯讀盤點，不要修改評分邏輯、YAML 權重、候選名單或刪除歷史資料。依交接文件檢查資料夾結構、真實 config 是否存在、FinMind token 是否已設定但不要顯示 token、prediction_audit.sqlite3 是否保留、四份正式 Notebook 的 Jupyter kernel、Python 套件與相對路徑。

接著在新電腦建立獨立 Python 3.12 環境，安裝兩份 requirements，註冊 pytorch-gpu 核心。若是舊版資料夾升級，先執行 migrate_report_layout.py；再執行 daily_pipeline.py --dry-run 與 prediction_audit.py status。以上通過後，執行一次完整 daily_pipeline.py，檢查最新 summary.json、TW/US 第一與第二階段 Excel，以及驗證資料庫筆數。確認全部成功後，再建立每天台北時間 20:00 的 Windows 工作排程；排程使用新環境的 python.exe、指向 daily_pipeline.py，開啟錯過後執行、喚醒、失敗重試，並禁止重疊執行。

最後請回報：使用的 Python 路徑與版本、套件或 kernel 缺漏、dry-run 結果、完整流程四階段狀態、最新 summary.json 路徑、報表路徑、驗證資料庫統計、排程名稱與下次執行時間，以及舊電腦排程是否已停用。若發現舊電腦絕對路徑，只修改真正的程式或設定來源，不要把 Notebook 的舊執行輸出誤判為程式綁死路徑。
```

## 13. 最近一次已驗證基準

舊電腦最近一次完整紀錄為 2026-09-04 台北時間 20:00：

- 台股第一階段：成功。
- 台股第二階段：成功，封存 132 筆預測、45 個指標。
- 美股第一階段：成功。
- 美股第二階段：成功，封存 96 筆預測、28 個指標。
- 整體 `status`：`ok`。
- 當次只有台股 `3088.TWO` 在舊預測結算時出現價格資料不可用警告，不影響四階段完成。

新電腦應以這份紀錄作為移機前基準，但不要求每次預測筆數完全相同，因為候選名單、市場交易日、資料供應狀態與執行日期都會改變。

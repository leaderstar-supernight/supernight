# WHID GitHub 版本管理

## 電腦分工

- 開發機：修改、測試與推送程式碼；不執行正式每日排程。
- 正式機：每天執行 WHID 與監控；只在確認版本後更新程式碼。
- GitHub `main`：兩台電腦共同使用的穩定程式版本。

## 標準更新流程

1. 開發機修改程式並完成 `dry-run` 或必要測試。
2. 開發機檢查差異，確認沒有 token、報表或驗證資料庫。
3. 開發機提交並推送到 GitHub。
4. 正式機確認當下沒有執行 `daily_pipeline.py`。
5. 正式機拉取 GitHub `main`。
6. 正式機執行 `daily_pipeline.py --dry-run`。
7. 驗證成功後才讓每天 20:00 的正式流程使用新版。

GitHub 不會讓兩台電腦即時自動同步。開發機必須 push，正式機必須 pull。正式機不建議在每晚流程前自動拉取未驗證版本。

## 會進入 GitHub

- 第一階段 Notebook：`Stock_Valuation_ReDesgin_TW.ipynb`、`Stock_Valuation_ReDesgin_US.ipynb`
- 第二階段程式與 Notebook：`Stock_price_prediction/*.py`、`Stock_price_prediction/*.ipynb`
- 第二階段設定：`Stock_price_prediction/config/*.yaml`
- requirements 與說明文件
- `config/*.example.yaml` 範本

## 不會進入 GitHub

- `.venv/`、`venv/`、`env/`
- `reports/`
- `Stock_price_prediction/reports/`
- `__pycache__/`、`.ipynb_checkpoints/`
- 專案根目錄 `config/*.yaml` 真實設定
- `prediction_audit.sqlite3` 與其他驗證資料
- Excel、Word 暫存檔

真實設定可能含 FinMind token。不得用強制加入方式繞過 `.gitignore`，也不要把 token 放進 Notebook、說明文件、commit 訊息或 Codex 對話。

## 正式資料的保存原則

GitHub 只傳遞程式碼與公開設定範本。正式機的真實 YAML、每日報表、快取及 `prediction_audit.sqlite3` 留在正式機本地，不由開發機覆蓋。第一次移機可以完整複製；完成交接後只透過 Git 更新程式碼。

新電腦完整安裝、排程及驗證方式請閱讀 `WINDOWS_移機交接說明.md`。

報表版型或路徑更新後，正式機先 pull，再執行 `python migrate_report_layout.py`。這個工具可重複執行，會保留驗證資料庫並更新其中的舊報表路徑。

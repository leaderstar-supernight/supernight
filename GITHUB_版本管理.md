# GitHub 版本管理

此專案建議使用 GitHub private repository 保存程式碼與文件，報表與快取保留在本機或雲端硬碟。

## 會進 GitHub

- 第一階段 Notebook：`Stock_Valuation_ReDesgin_TW.ipynb`、`Stock_Valuation_ReDesgin_US.ipynb`
- 第二階段程式與 Notebook：`Stock_price_prediction/*.py`、`Stock_price_prediction/*.ipynb`
- 第二階段設定：`Stock_price_prediction/config/*.yaml`
- requirements 與說明文件
- `config/*.example.yaml` 範本

## 不會進 GitHub

- `.venv/`
- `reports/`
- `Stock_price_prediction/reports/`
- `__pycache__/`
- `.ipynb_checkpoints/`
- root `config/*.yaml` 真實設定檔
- Excel 暫存檔 `~$*.xlsx`

## 新機器使用方式

1. 從 GitHub clone 專案。
2. 依照 `config/*.example.yaml` 建立本機真實設定，例如 `config/TW.yaml`。
3. 在 `Stock_price_prediction` 建立 `.venv` 並安裝 requirements。
4. 執行 `.venv/bin/python daily_pipeline.py --dry-run` 確認流程入口。


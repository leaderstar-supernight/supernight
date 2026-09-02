# Mac 流程測試筆記

本資料夾已建立本機虛擬環境：

```bash
cd /Volumes/ZhenWu/Stock_Valuation/Stock_price_prediction
source .venv/bin/activate
```

已安裝 `requirements-timing.txt`、`requirements-timing-US.txt` 與 `ipykernel`。Notebook 需要的 `pytorch-gpu` kernel 已註冊到此 `.venv`。

## 快速檢查

```bash
.venv/bin/python daily_pipeline.py --dry-run
```

這只檢查台股／美股 Notebook 路徑、pipeline 串接與封存模組，不下載行情、不執行完整 Notebook。

## 單檔 Smoke Test

美股：

```bash
.venv/bin/python - <<'PY'
import pandas as pd
from timing_us import run
candidate = pd.DataFrame([{'代號': 'AAPL', '測試來源': 'Mac smoke test'}])
result = run(candidate_frame=candidate, provenance={'來源': 'Mac smoke test', 'WHID評估日期': None}, write=True)
print(result['simple'].to_string(index=False))
print(result['paths'])
PY
```

台股：

```bash
.venv/bin/python - <<'PY'
import pandas as pd
from timing_tw import run
candidate = pd.DataFrame([{'代號': '2330.TW', '測試來源': 'Mac smoke test'}])
result = run(candidate_frame=candidate, provenance={'來源': 'Mac smoke test', 'WHID評估日期': None}, write=True)
print(result['simple'].to_string(index=False))
print(result['paths'])
PY
```

## 完整每日流程

```bash
.venv/bin/python daily_pipeline.py --market ALL
```

也可以只跑單一市場：

```bash
.venv/bin/python daily_pipeline.py --market TW
.venv/bin/python daily_pipeline.py --market US
```

完整流程會依序執行第一階段 WHID Notebook 與第二階段 Notebook，時間會比 smoke test 長很多。

## 已知 Mac 注意事項

- 目前 Mac 沒有 `libomp.dylib`，`xgboost` 在部分歷史評估會回報 OpenMP runtime 缺漏。程式已改成將此狀態寫入 `AIStatus`，不讓整個流程中斷。
- 若要完整啟用 XGBoost，安裝 Homebrew 後執行 `brew install libomp`。
- macOS 內建 Python 使用 LibreSSL，執行時可能出現 `urllib3 NotOpenSSLWarning`。目前 smoke test 可正常下載資料與輸出報表。

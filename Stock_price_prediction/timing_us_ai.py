"""Independent US calibrated three-class AI research models; never changes trading advice."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    'return_1', 'return_5', 'return_20', 'return_60', 'distance_20',
    'distance_60', 'rsi', 'atr_pct', 'volume_ratio', 'volatility_20',
    'range_position',
]
CLASSES = ['down_first', 'neutral', 'up_first']
CLASS_ZH = {'down_first': '下行先觸', 'neutral': '盤整', 'up_first': '上行先觸'}


def atr_barrier_labels(prices, feature_frame, cfg):
    """Label which ATR barrier is touched first; same-day double touch is down."""
    horizon = int(cfg['horizon'])
    atr = feature_frame['atr'].reindex(prices.index)
    result = pd.DataFrame(index=prices.index, columns=['target', 'maturity'], dtype=object)
    for pos in range(len(prices) - horizon):
        close = prices.Close.iloc[pos]
        width = atr.iloc[pos]
        if not np.isfinite(close) or close <= 0 or not np.isfinite(width) or width <= 0:
            continue
        up_level = close + float(cfg['up_atr']) * width
        down_level = close - float(cfg['down_atr']) * width
        target = 'neutral'
        for step in range(1, horizon + 1):
            row = prices.iloc[pos + step]
            down_hit = np.isfinite(row.Low) and row.Low <= down_level
            up_hit = np.isfinite(row.High) and row.High >= up_level
            if down_hit:
                target = 'down_first'
                break
            if up_hit:
                target = 'up_first'
                break
        result.iat[pos, 0] = target
        result.iat[pos, 1] = prices.index[pos + horizon]
    result['target'] = pd.Categorical(result.target, categories=CLASSES)
    result['maturity'] = pd.to_datetime(result.maturity)
    return result


def _sequences(features, sequence_length):
    values = features.to_numpy(dtype=np.float32)
    output = {}
    for pos in range(sequence_length - 1, len(features)):
        block = values[pos - sequence_length + 1:pos + 1]
        if np.isfinite(block).all():
            output[features.index[pos]] = block
    return output


def samples(prices, features, cfg):
    x = features[FEATURES].replace([np.inf, -np.inf], np.nan)
    labels = atr_barrier_labels(prices, features, cfg)
    sequence_map = _sequences(x, int(cfg['sequence_length']))
    ready = pd.Series(False, index=x.index)
    if sequence_map:
        ready.loc[list(sequence_map)] = True
    data = x.join(labels)
    data = data.loc[ready & data.target.notna() & data.maturity.notna()].copy()
    data['target_id'] = data.target.map({name: i for i, name in enumerate(CLASSES)}).astype(int)
    latest = x.tail(1)
    if latest.empty or latest.isna().any().any() or latest.index[0] not in sequence_map:
        latest = pd.DataFrame(columns=FEATURES)
    return data, latest, sequence_map


def partitions(data, cfg):
    final_n = int(cfg['final_test'])
    required = int(cfg['min_train']) + int(cfg['calibration_size']) + 2 * int(cfg['horizon']) + final_n
    if len(data) < required:
        return []
    final = data.iloc[-final_n:]
    pre = data.loc[data.maturity < final.index[0]]
    folds = []
    first = int(cfg['min_train']) + int(cfg['calibration_size']) + 2 * int(cfg['horizon'])
    for start in range(first, len(pre), int(cfg['validation_size'])):
        test = pre.iloc[start:start + int(cfg['validation_size'])]
        if len(test) < 20:
            continue
        train = pre.loc[pre.maturity < test.index[0]]
        folds.append(('walk_forward', len(folds) + 1, train, test))
    folds.append(('final_test', 0, pre, final))
    return folds


def _enough(y, minimum):
    counts = pd.Series(y).value_counts()
    return all(int(counts.get(k, 0)) >= int(minimum) for k in range(len(CLASSES)))


def _class_weights(y):
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    return len(y) / (len(CLASSES) * np.maximum(counts, 1.0))


class _TorchLSTM:
    def __init__(self, cfg, input_size):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError('LSTM需要PyTorch；請先執行Notebook安裝套件格') from exc
        self.torch = torch
        torch.manual_seed(int(cfg['seed']))
        np.random.seed(int(cfg['seed']))
        hidden = int(cfg['lstm_hidden'])
        dropout = float(cfg['lstm_dropout'])

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
                self.dropout = nn.Dropout(dropout)
                self.output = nn.Linear(hidden, len(CLASSES))

            def forward(self, x):
                values, _ = self.lstm(x)
                return self.output(self.dropout(values[:, -1, :]))

        self.net = Network()
        self.cfg = cfg
        self.mean = None
        self.scale = None

    def fit(self, x, y):
        torch = self.torch
        from torch.utils.data import DataLoader, TensorDataset
        n = len(y)
        val_n = min(int(self.cfg['lstm_validation_size']), max(1, n // 5))
        split = n - val_n - int(self.cfg['horizon'])
        if split < int(self.cfg['min_train']) // 2:
            raise ValueError('LSTM時間隔離後訓練樣本不足')
        x_train, y_train = x[:split], y[:split]
        x_val, y_val = x[n - val_n:], y[n - val_n:]
        self.mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
        self.scale = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        normalize = lambda value: (value - self.mean) / self.scale
        train_ds = TensorDataset(torch.tensor(normalize(x_train), dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_x = torch.tensor(normalize(x_val), dtype=torch.float32)
        val_y = torch.tensor(y_val, dtype=torch.long)
        loader = DataLoader(train_ds, batch_size=int(self.cfg['lstm_batch_size']), shuffle=False)
        weights = torch.tensor(_class_weights(y_train), dtype=torch.float32)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=float(self.cfg['lstm_learning_rate']))
        best, best_state, stale = float('inf'), None, 0
        for _ in range(int(self.cfg['lstm_epochs'])):
            self.net.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                optimizer.step()
            self.net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.net(val_x), val_y).item())
            if val_loss < best - 1e-5:
                best = val_loss
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= int(self.cfg['lstm_patience']):
                    break
        if best_state is None:
            raise RuntimeError('LSTM未完成訓練')
        self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, x):
        if self.mean is None:
            raise RuntimeError('LSTM尚未訓練')
        values = (x - self.mean) / self.scale
        self.net.eval()
        with self.torch.no_grad():
            logits = self.net(self.torch.tensor(values, dtype=self.torch.float32))
            return self.torch.softmax(logits, dim=1).cpu().numpy()


def _base_model(model_name, cfg, input_size):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if model_name == 'logistic':
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.1, max_iter=1500, class_weight='balanced', random_state=int(cfg['seed'])))
    if model_name == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=120, max_depth=2, min_child_weight=10, learning_rate=.03,
            subsample=.8, colsample_bytree=.8, objective='multi:softprob',
            num_class=len(CLASSES), eval_metric='mlogloss', random_state=int(cfg['seed']), n_jobs=1)
    if model_name == 'lstm':
        return _TorchLSTM(cfg, input_size)
    raise ValueError(f'未知模型：{model_name}')


def _matrix(frame, model_name, sequence_map):
    if model_name == 'lstm':
        return np.stack([sequence_map[day] for day in frame.index])
    return frame[FEATURES]


def fit_calibrated(train, infer, model_name, cfg, sequence_map):
    """Fit on old data, calibrate on a later block, and infer without leakage."""
    from sklearn.linear_model import LogisticRegression
    cal_n = int(cfg['calibration_size'])
    if len(train) < cal_n + int(cfg['min_train']):
        raise ValueError('訓練／校準樣本不足')
    cal = train.iloc[-cal_n:]
    fit = train.loc[train.maturity < cal.index[0]]
    if len(fit) < int(cfg['min_train']):
        raise ValueError('時間隔離後訓練樣本不足')
    if not _enough(fit.target_id, cfg['min_class_count']) or not _enough(cal.target_id, cfg['min_class_count']):
        raise ValueError('上行／下行／盤整樣本不足；不輸出假機率')
    model = _base_model(model_name, cfg, len(FEATURES))
    x_fit = _matrix(fit, model_name, sequence_map)
    y_fit = fit.target_id.to_numpy(dtype=int)
    if model_name == 'xgboost':
        weights = _class_weights(y_fit)
        model.fit(x_fit, y_fit, sample_weight=weights[y_fit])
    else:
        model.fit(x_fit, y_fit)
    cal_raw = np.clip(model.predict_proba(_matrix(cal, model_name, sequence_map)), 1e-7, 1)
    calibrator = LogisticRegression(C=1., max_iter=1500, random_state=int(cfg['seed']))
    calibrator.fit(np.log(cal_raw), cal.target_id.astype(int))
    raw = np.clip(model.predict_proba(_matrix(infer, model_name, sequence_map)), 1e-7, 1)
    probability = calibrator.predict_proba(np.log(raw))
    # Calibrator classes can only differ if validation was bypassed; normalize defensively.
    aligned = np.zeros((len(infer), len(CLASSES)))
    aligned[:, calibrator.classes_.astype(int)] = probability
    aligned /= aligned.sum(axis=1, keepdims=True)
    baseline = train.target_id.value_counts(normalize=True).reindex(range(len(CLASSES)), fill_value=0).to_numpy()
    meta = {
        'fit_end': str(fit.index[-1].date()),
        'calibration_start': str(cal.index[0].date()),
        'calibration_end': str(cal.index[-1].date()),
        'fit_count': len(fit), 'calibration_count': len(cal),
    }
    return aligned, baseline, meta


def score(actual, probability, baseline):
    from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score
    from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
    y = np.asarray(actual, dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = p.argmax(axis=1)
    one_hot = np.eye(len(CLASSES))[y]
    base = np.tile(np.asarray(baseline, dtype=float), (len(y), 1))
    selected = p.max(axis=1) >= .5
    result = {
        'samples': len(y), 'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, labels=range(len(CLASSES)), average='macro', zero_division=0)),
        'log_loss': float(log_loss(y, p, labels=range(len(CLASSES)))),
        'baseline_log_loss': float(log_loss(y, base, labels=range(len(CLASSES)))),
        'brier': float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        'baseline_brier': float(np.mean(np.sum((base - one_hot) ** 2, axis=1))),
        'confident_50pct_count': int(selected.sum()),
        'confident_50pct_accuracy': float((pred[selected] == y[selected]).mean()) if selected.any() else np.nan,
    }
    result['macro_average_precision'] = (
        float(average_precision_score(one_hot, p, average='macro'))
        if len(np.unique(y)) == len(CLASSES) else np.nan)
    precision = precision_score(y, pred, labels=range(len(CLASSES)), average=None, zero_division=0)
    recall = recall_score(y, pred, labels=range(len(CLASSES)), average=None, zero_division=0)
    for idx, name in enumerate(CLASSES):
        result[f'{name}_count'] = int((y == idx).sum())
        result[f'{name}_rate'] = float((y == idx).mean())
        result[f'{name}_precision'] = float(precision[idx])
        result[f'{name}_recall'] = float(recall[idx])
    return result


def _long_predictions(index, actual, probability, baseline, period, fold, model):
    rows = []
    for class_id, class_name in enumerate(CLASSES):
        rows.append(pd.DataFrame({
            'date': index, 'actual_class': [CLASSES[int(v)] for v in actual],
            'class': class_name, 'actual': (np.asarray(actual) == class_id).astype(int),
            'probability': probability[:, class_id], 'baseline_probability': baseline[class_id],
            'period': period, 'fold': fold, 'model': model,
        }))
    return pd.concat(rows, ignore_index=True)


def calibration_bins(predictions):
    rows = []
    if predictions.empty:
        return pd.DataFrame()
    for (period, model, class_name), group in predictions.groupby(['period', 'model', 'class']):
        for lo in np.arange(0, 1, .1):
            hi = round(float(lo + .1), 2)
            selected = group.loc[(group.probability >= lo) & ((group.probability < hi) if hi < 1 else (group.probability <= 1))]
            if selected.empty:
                continue
            rows.append({
                'period': period, 'model': model, 'class': class_name,
                'bin': f'{lo:.0%}–{hi:.0%}', 'count': len(selected),
                'mean_probability': float(selected.probability.mean()),
                'actual_rate': float(selected.actual.mean()),
            })
    return pd.DataFrame(rows)


def run_ai(prices, feature_frame, config):
    cfg = config['ai']
    empty = {'latest': pd.DataFrame(), 'metrics': pd.DataFrame(), 'predictions': pd.DataFrame(),
             'calibration': pd.DataFrame(), 'status': '停用', 'errors': []}
    if not cfg['enabled']:
        return empty
    if prices is None:
        empty['status'] = '缺完整還原價格，AI停用'
        return empty
    try:
        import sklearn  # noqa: F401
    except ImportError:
        empty['status'] = '未安裝scikit-learn；規則分析仍可用'
        return empty
    data, latest, sequence_map = samples(prices, feature_frame, cfg)
    if latest.empty:
        empty['status'] = '最新日特徵不足，未使用較舊日期代替'
        return empty
    parts = partitions(data, cfg)
    errors, prediction_frames, metrics, now_rows = [], [], [], []
    ensemble_models = list(cfg.get('ensemble_models', ['logistic', 'xgboost']))
    for period, fold, train, test in parts:
        fold_probabilities = {}
        for model_name in cfg['models']:
            if model_name == 'lstm' and period == 'walk_forward' and not cfg.get('lstm_walk_forward', False):
                continue
            try:
                probability, baseline, meta = fit_calibrated(train, test, model_name, cfg, sequence_map)
                fold_probabilities[model_name] = probability
                prediction_frames.append(_long_predictions(
                    test.index, test.target_id.to_numpy(), probability, baseline, period, fold, model_name))
                metrics.append({'period': period, 'fold': fold, 'model': model_name,
                    'test_start': str(test.index[0].date()), 'test_end': str(test.index[-1].date()),
                    **meta, **score(test.target_id, probability, baseline)})
            except (ImportError, ValueError, RuntimeError) as exc:
                errors.append(f'{period}/{fold}/{model_name}: {type(exc).__name__}: {str(exc)[:180]}')
        if all(name in fold_probabilities for name in ensemble_models):
            ensemble = np.mean([fold_probabilities[name] for name in ensemble_models], axis=0)
            baseline = train.target_id.value_counts(normalize=True).reindex(range(len(CLASSES)), fill_value=0).to_numpy()
            prediction_frames.append(_long_predictions(
                test.index, test.target_id.to_numpy(), ensemble, baseline, period, fold, 'ensemble'))
            metrics.append({'period': period, 'fold': fold, 'model': 'ensemble',
                'test_start': str(test.index[0].date()), 'test_end': str(test.index[-1].date()),
                **score(test.target_id, ensemble, baseline)})
    mature = data.loc[data.maturity <= latest.index[0]]
    signal_close = float(prices.Close.reindex(latest.index).iloc[0])
    signal_atr = float(feature_frame['atr'].reindex(latest.index).iloc[0])
    upper_barrier = signal_close + float(cfg['up_atr']) * signal_atr
    lower_barrier = signal_close - float(cfg['down_atr']) * signal_atr
    latest_probabilities = {}
    for model_name in cfg['models']:
        try:
            probability, baseline, meta = fit_calibrated(mature, latest, model_name, cfg, sequence_map)
            latest_probabilities[model_name] = probability[0]
            for idx, class_name in enumerate(CLASSES):
                now_rows.append({
                    'prediction_date': str(latest.index[0].date()), 'model': model_name,
                    'class': class_name, 'class_zh': CLASS_ZH[class_name],
                    'probability': float(probability[0, idx]), 'horizon': int(cfg['horizon']),
                    'up_atr': float(cfg['up_atr']), 'down_atr': float(cfg['down_atr']),
                    'signal_close': signal_close, 'signal_atr': signal_atr,
                    'upper_barrier': upper_barrier, 'lower_barrier': lower_barrier,
                    'ensemble_role': 'baseline' if model_name in ensemble_models else 'challenger',
                    'status': '研究參考，不參與交易建議', **meta,
                })
        except (ImportError, ValueError, RuntimeError) as exc:
            errors.append(f'latest/{model_name}: {type(exc).__name__}: {str(exc)[:180]}')
    if all(name in latest_probabilities for name in ensemble_models):
        ensemble = np.mean([latest_probabilities[name] for name in ensemble_models], axis=0)
        for idx, class_name in enumerate(CLASSES):
            now_rows.append({
                'prediction_date': str(latest.index[0].date()), 'model': 'ensemble',
                'class': class_name, 'class_zh': CLASS_ZH[class_name],
                'probability': float(ensemble[idx]), 'horizon': int(cfg['horizon']),
                'up_atr': float(cfg['up_atr']), 'down_atr': float(cfg['down_atr']),
                'signal_close': signal_close, 'signal_atr': signal_atr,
                'upper_barrier': upper_barrier, 'lower_barrier': lower_barrier,
                'ensemble_role': 'production_baseline',
                'status': 'Logistic與XGBoost等權平均；研究參考，不參與交易建議',
            })
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    if any(row['model'] == 'ensemble' for row in now_rows):
        lstm_ready = any(row['model'] == 'lstm' for row in now_rows)
        status = ('三分類研究機率已產生；LSTM挑戰模型已完成；不參與交易建議' if lstm_ready
                  else '三分類基準機率已產生；LSTM挑戰模型未完成，詳見AIStatus；不參與交易建議')
    elif now_rows:
        status = '個別模型有結果但基準集成未成立；機率不顯示'
    else:
        latest_errors = [item for item in errors if item.startswith('latest/')]
        if any('ImportError' in item for item in latest_errors):
            status = '模型套件缺漏，未產生基準集成機率'
        elif latest_errors:
            status = '三分類訓練／校準樣本不足，機率留白'
        else:
            status = '模型未完成，請查看AIStatus原因'
        if metrics:
            status += '；已有部分歷史評估'
    return {
        'latest': pd.DataFrame(now_rows), 'metrics': pd.DataFrame(metrics),
        'predictions': predictions, 'calibration': calibration_bins(predictions),
        'errors': errors, 'status': status,
    }

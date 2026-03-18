"""
벤치마크 비교 스크립트

사용법:
  python scripts/benchmark.py            # 벤치마크 모델 학습 + 평가
  python scripts/benchmark.py --predict  # 저장된 벤치마크 모델로 평가만 실행
"""
import sys
import os
import json
import pickle
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import get_device, make_dataloader, train_model
from src.evaluate import (
    predict_model, backtest_by_ticker, backtest_buy_and_hold,
    get_risk_free_rate, find_optimal_threshold,
    direction_accuracy, safe_auc,
)
from src.contime import build_contime_lstm_model
from src.benchmarks import build_vanilla_lstm, build_contime_gru, delong_roc_test
from src.data.pipeline import clean_numeric_data
from src.utils import get_project_root
from src.config import (
    COMMISSION, DEFAULT_DROPOUT,
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_PATIENCE,
    MIN_TRADES_RATIO,
)

CHECKPOINTS  = Path(get_project_root()) / "output" / "checkpoints"
CONTIME_NAME = "best_contime"


# ── 데이터 로드 ────────────────────────────────────────────────

def _find_latest_pkl():
    processed_dir = Path(get_project_root()) / "output" / "processed"
    pkls = list(processed_dir.glob("*_processed.pkl"))
    if not pkls:
        raise FileNotFoundError("전처리 데이터 없음. 먼저 prepare.py 실행 필요")
    return max(pkls, key=lambda p: p.stat().st_mtime)


def load_processed_data():
    pkl_path = _find_latest_pkl()
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    raw_path = (pkl_path.parents[1] / "raw" /
                pkl_path.name.replace("_processed.pkl", "_data.csv"))
    if raw_path.exists():
        import pandas as pd
        data['data'] = pd.read_csv(raw_path)

    meta_path = pkl_path.with_name(
        pkl_path.name.replace("_processed.pkl", "_metadata.json"))
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    print(f"데이터: {metadata.get('tickers', '?')}")
    print(f"기간: {metadata.get('start_date','?')} ~ {metadata.get('end_date','?')}")
    return data, metadata


# ── ConTime-LSTM 로드 ──────────────────────────────────────────

def load_contime_lstm(data, device):
    config_path = CHECKPOINTS / f"{CONTIME_NAME}_config.json"
    model_path  = CHECKPOINTS / f"{CONTIME_NAME}.pt"
    meta_path   = CHECKPOINTS / f"{CONTIME_NAME}_meta.json"

    for p in [config_path, model_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"파일 없음: {p}\nrun.py 먼저 실행 필요")

    with open(config_path) as f: config = json.load(f)
    with open(meta_path)   as f: meta   = json.load(f)

    x_test = data['x_test']
    model = build_contime_lstm_model(
        seq_len=x_test.shape[1],
        num_features=x_test.shape[2],
        hidden_dim=int(config['hidden_dim']),
        dropout_rate=float(config['dropout_rate']),
        dt=float(config['dt']),
        ode_steps=int(config['ode_steps']),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, float(meta['best_threshold']), config


# ── 공통 유틸 ──────────────────────────────────────────────────

def _test_arrays(data):
    x      = clean_numeric_data(data['x_test'])
    ticker = np.asarray(data['ticker_test'], dtype=np.int64)
    td     = np.asarray(data['time_diffs_test'], dtype=np.float32)
    y      = np.asarray(data['y_test']).flatten()
    return x, ticker, td, y


def _get_preds(model, data, device):
    x, ticker, td, y = _test_arrays(data)
    preds, _ = predict_model(model, x, td, device=device)
    return preds[:, -1, 0], y, ticker


def _find_threshold(model, data, device, rfr):
    x_val  = clean_numeric_data(data['x_val'])
    ticker = np.asarray(data['ticker_val'], dtype=np.int64)
    td     = np.asarray(data['time_diffs_val'], dtype=np.float32)
    y      = np.asarray(data['y_val']).flatten()

    preds, _ = predict_model(model, x_val, td, device=device)
    y_pred   = preds[:, -1, 0]

    total_opp  = len(y)
    min_trades = max(5, int(total_opp * MIN_TRADES_RATIO))
    threshold, _, _ = find_optimal_threshold(
        y_pred, y, ticker, rfr, min_trades,
        use_combined_score=True, total_opportunities=total_opp,
    )
    return threshold


def _run_backtest(preds, y, ticker_ids, threshold, rfr):
    return backtest_by_ticker(
        predictions=preds, actual_returns=y, ticker_ids=ticker_ids,
        threshold=threshold, commission=COMMISSION, risk_free_rate=rfr,
    )




# ── 벤치마크 모델 학습 ─────────────────────────────────────────

def _train_bench(model, data, config, device, value_weight):
    y_tr   = data['y_train']
    y_v    = data['y_val']
    ydt_tr = data.get('y_train_dt', np.zeros_like(y_tr))
    ydt_v  = data.get('y_val_dt',   np.zeros_like(y_v))

    x_tr = clean_numeric_data(data['x_train'])
    x_v  = clean_numeric_data(data['x_val'])

    train_loader = make_dataloader(
        x_tr, data['time_diffs_train'], y_tr, ydt_tr, DEFAULT_BATCH_SIZE, shuffle=True,
    )
    val_loader = make_dataloader(
        x_v, data['time_diffs_val'], y_v, ydt_v, DEFAULT_BATCH_SIZE, shuffle=False,
    )
    train_model(model, train_loader, val_loader, config,
                value_weight=value_weight,
                derivative_weight=1.0 - value_weight,
                device=device)
    return model


# ── 결과 출력 ──────────────────────────────────────────────────

def _print_table(results):
    print("\n" + "=" * 78)
    print("벤치마크 비교 (test set)")
    print("=" * 78)
    print(f"{'Model':<22} {'Return':>9} {'Sharpe':>8} {'DTW':>9} "
          f"{'TDI':>9} {'Trd.Acc':>9} {'AUC':>8}")
    print("-" * 78)
    for name, r in results.items():
        p = r['portfolio']
        print(f"{name:<22}"
              f" {p['total_return']:>9.4f}"
              f" {p.get('sharpe_ratio', float('nan')):>8.4f}"
              f" {p.get('dtw', float('nan')):>9.4f}"
              f" {p.get('tdi', float('nan')):>9.4f}"
              f" {r.get('trading_accuracy', float('nan')):>9.4f}"
              f" {r.get('auc', float('nan')):>8.4f}")
    print("=" * 78)


def _print_delong(delong_results):
    print("\n[DeLong AUC Test]  H0: AUC(ConTime-LSTM) == AUC(benchmark)")
    print(f"{'Comparison':<36} {'AUC_A':>7} {'AUC_B':>7} "
          f"{'Z':>8} {'p-value':>10}  Sig")
    print("-" * 78)
    for name, r in delong_results.items():
        p = r['p_value']
        sig = ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '')
        print(f"ConTime-LSTM vs {name:<21}"
              f" {r['auc_a']:>7.4f} {r['auc_b']:>7.4f}"
              f" {r['z']:>8.3f} {p:>10.4f}  {sig}")


# ── main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="벤치마크 모델 학습 및 비교")
    parser.add_argument('--predict', action='store_true',
                        help='저장된 벤치마크 모델로 평가만 실행')
    args = parser.parse_args()

    device = get_device()
    print(f"디바이스: {device}")

    data, metadata = load_processed_data()
    rfr = get_risk_free_rate(metadata.get('start_date'), metadata.get('end_date'))

    seq_len      = data['x_test'].shape[1]
    num_features = data['x_test'].shape[2]

    bench_config = {
        'epochs':       DEFAULT_EPOCHS,
        'patience':     DEFAULT_PATIENCE,
        'dropout_rate': DEFAULT_DROPOUT,
        'batch_size':   DEFAULT_BATCH_SIZE,
    }

    results   = {}
    all_preds = {}

    # ── 1. ConTime-LSTM (저장된 best model) ──────────────────────
    print("\n[1/4] ConTime-LSTM 로드...")
    lstm_model, threshold_lstm, best_config = load_contime_lstm(data, device)
    preds_lstm, y_test, ticker_test = _get_preds(lstm_model, data, device)

    bt = _run_backtest(preds_lstm, y_test, ticker_test, threshold_lstm, rfr)
    bt['trading_accuracy'] = direction_accuracy(preds_lstm, y_test)
    bt['auc']              = safe_auc(y_test, preds_lstm)
    results['ConTime-LSTM']    = bt
    all_preds['ConTime-LSTM']  = preds_lstm

    # ── 2. Buy & Hold ────────────────────────────────────────────
    print("\n[2/4] Buy & Hold...")
    _, ticker, _, y = _test_arrays(data)
    bt_bh = backtest_buy_and_hold(y, ticker, commission=COMMISSION, risk_free_rate=rfr)
    bt_bh['trading_accuracy']     = float(np.mean(y > 0))
    bt_bh['auc']                  = float('nan')
    bt_bh['portfolio']['dtw']     = float('nan')
    bt_bh['portfolio']['tdi']     = float('nan')
    results['Buy & Hold'] = bt_bh

    # ── 3. Vanilla LSTM ──────────────────────────────────────────
    print("\n[3/4] Vanilla LSTM...")
    vanilla_path = CHECKPOINTS / "bench_vanilla_lstm.pt"
    vanilla = build_vanilla_lstm(
        seq_len, num_features,
        hidden_dim=int(best_config['hidden_dim']),
    ).to(device)

    if args.predict and vanilla_path.exists():
        vanilla.load_state_dict(torch.load(vanilla_path, map_location=device))
        print("  저장된 모델 로드")
    else:
        _train_bench(vanilla, data, bench_config, device, value_weight=1.0)
        torch.save(vanilla.state_dict(), vanilla_path)
        print(f"  모델 저장: {vanilla_path}")

    threshold_vanilla = _find_threshold(vanilla, data, device, rfr)
    preds_vanilla, _, _ = _get_preds(vanilla, data, device)
    bt = _run_backtest(preds_vanilla, y_test, ticker_test, threshold_vanilla, rfr)
    bt['trading_accuracy']    = direction_accuracy(preds_vanilla, y_test)
    bt['auc']                 = safe_auc(y_test, preds_vanilla)
    results['Vanilla LSTM']   = bt
    all_preds['Vanilla LSTM'] = preds_vanilla

    # ── 4. ConTime-GRU ───────────────────────────────────────────
    print("\n[4/4] ConTime-GRU...")
    gru_path = CHECKPOINTS / "bench_contime_gru.pt"
    gru = build_contime_gru(
        seq_len, num_features,
        hidden_dim=int(best_config['hidden_dim']),
        dt=float(best_config['dt']),
        ode_steps=int(best_config['ode_steps']),
    ).to(device)

    if args.predict and gru_path.exists():
        gru.load_state_dict(torch.load(gru_path, map_location=device))
        print("  저장된 모델 로드")
    else:
        vw = float(best_config.get('value_weight', 0.8))
        _train_bench(gru, data, bench_config, device, value_weight=vw)
        torch.save(gru.state_dict(), gru_path)
        print(f"  모델 저장: {gru_path}")

    threshold_gru = _find_threshold(gru, data, device, rfr)
    preds_gru, _, _ = _get_preds(gru, data, device)
    bt = _run_backtest(preds_gru, y_test, ticker_test, threshold_gru, rfr)
    bt['trading_accuracy']   = direction_accuracy(preds_gru, y_test)
    bt['auc']                = safe_auc(y_test, preds_gru)
    results['ConTime-GRU']   = bt
    all_preds['ConTime-GRU'] = preds_gru

    # ── 결과 출력 ─────────────────────────────────────────────────
    _print_table(results)

    y_binary = (y_test > 0).astype(int)
    delong_results = {}
    for name in ['Vanilla LSTM', 'ConTime-GRU']:
        z, p, auc_a, auc_b = delong_roc_test(
            y_binary, all_preds['ConTime-LSTM'], all_preds[name]
        )
        delong_results[name] = {'z': z, 'p_value': p, 'auc_a': auc_a, 'auc_b': auc_b}
    _print_delong(delong_results)


if __name__ == "__main__":
    main()

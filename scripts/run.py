"""
주식 예측 모델 실행 스크립트

사용법:
  python scripts/run.py            # 그리드 서치 + 학습
  python scripts/run.py --predict  # 저장된 최적 모델로 예측만 실행
"""
import sys
import os
import argparse
import json
import pickle
import numpy as np
import torch
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import run_optimization_pipeline, get_device
from src.utils import get_project_root
from src.contime import build_contime_lstm_model
from src.evaluate import predict_model, backtest_by_ticker
from src.config import DEFAULT_TICKERS, COMMISSION


# ── 저장 경로 상수 ──────────────────────────────
MODEL_NAME   = 'best_contime'
OUTPUT_FILE  = 'grid_search_results.pkl'
METRIC       = 'combined_score'


def load_processed_data(tickers):
    """전처리된 데이터와 메타 정보를 로드합니다."""
    output_dir    = get_project_root() / "output"
    processed_dir = output_dir / "processed"

    if tickers:
        tickers_path   = tickers.replace(',', '_') if ',' in tickers else tickers
        processed_path = processed_dir / f"{tickers_path}_processed.pkl"
    else:
        pkls = list(processed_dir.glob("*_processed.pkl"))
        if not pkls:
            print("오류: 전처리된 데이터 파일이 없습니다. prepare.py를 먼저 실행하세요.")
            return None, None, None
        processed_path = max(pkls, key=lambda p: p.stat().st_mtime)
        tickers_path   = processed_path.stem.replace("_processed", "")

    if not processed_path.exists():
        print(f"오류: 전처리된 데이터 파일이 없습니다: {processed_path}")
        return None, None, None

    with open(processed_path, 'rb') as f:
        processed_data = pickle.load(f)

    raw_data_path = output_dir / "raw" / f"{tickers_path}_data.csv"
    if raw_data_path.exists():
        processed_data['data'] = pd.read_csv(raw_data_path)

    encoder_info = {}
    encoder_path = processed_dir / f"{tickers_path}_encoder_info.json"
    if encoder_path.exists():
        with open(encoder_path, 'r') as f:
            encoder_info = json.load(f)

    metadata = {}
    metadata_path = processed_dir / f"{tickers_path}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"\n데이터셋: {', '.join(metadata.get('tickers', []))}")
        print(f"기간: {metadata.get('start_date', '')} ~ {metadata.get('end_date', '')}")
        print(f"특성 수: {metadata.get('feature_count', '')}, 윈도우: {metadata.get('window_size', '')}")

    return processed_data, encoder_info, metadata


def run_predict(processed_data):
    """저장된 최적 모델로 테스트 세트 예측 및 백테스트 실행."""
    results_dir = Path(get_project_root()) / "output" / "checkpoints"
    model_path  = results_dir / f"{MODEL_NAME}.pt"
    config_path = results_dir / f"{MODEL_NAME}_config.json"
    meta_path   = results_dir / f"{MODEL_NAME}_meta.json"

    for p in [model_path, config_path, meta_path]:
        if not p.exists():
            print(f"파일 없음: {p}\n먼저 학습을 실행하세요 (--predict 없이).")
            return

    with open(config_path, 'r') as f:
        config = json.load(f)
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    threshold = meta['best_threshold']
    x_test = processed_data['x_test']
    device = get_device()

    model = build_contime_lstm_model(
        seq_len=x_test.shape[1],
        num_features=x_test.shape[2],
        hidden_dim=int(config['hidden_dim']),
        dropout_rate=float(config['dropout_rate']),
        dt=float(config['dt']),
        ode_steps=int(config['ode_steps']),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"모델 로드 완료: {model_path}")

    ticker_test = np.asarray(processed_data['ticker_test'], dtype=np.int64)
    time_diffs  = np.asarray(processed_data['time_diffs_test'], dtype=np.float32)

    preds, _ = predict_model(model, x_test, time_diffs, device=device)
    y_pred = preds[:, -1, 0]

    result = backtest_by_ticker(
        predictions=y_pred,
        actual_returns=processed_data['y_test'].flatten(),
        ticker_ids=ticker_test,
        threshold=threshold,
        commission=COMMISSION,
    )

    print(f"\n===== 예측 결과 (threshold={threshold:.4f}) =====")
    print(f"총 수익률  : {result['portfolio']['total_return']:.4f}")
    print(f"샤프 비율  : {result['portfolio']['sharpe_ratio']:.4f}")
    print(f"최대 낙폭  : {result['portfolio']['max_drawdown']:.4f}")
    print(f"거래 횟수  : {len(result['portfolio'].get('trades', []))}")
    print(f"승률       : {result['portfolio'].get('trade_win_rate', 0):.2%}")
    print(f"DTW        : {result['portfolio'].get('dtw', 0):.4f}")
    print(f"TDI        : {result['portfolio'].get('tdi', 0):.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description="주식 예측 모델 학습 및 예측")
    parser.add_argument('--predict', action='store_true',
                        help='학습 없이 저장된 최적 모델로 예측만 실행')
    args = parser.parse_args()

    processed_data, _, metadata = load_processed_data(DEFAULT_TICKERS)
    if processed_data is None:
        return

    np.random.seed(42)

    if args.predict:
        run_predict(processed_data)
        return

    # ── 학습 모드 ───────────────────────────────
    print("\n모델 최적화 시작...")
    results_dir = Path(get_project_root()) / "output" / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)

    optimization_results = run_optimization_pipeline(
        data_dict=processed_data,
        metric=METRIC,
        output_path=results_dir / OUTPUT_FILE,
        save=True,
        model_output=results_dir / MODEL_NAME,
        run_visualizations=True,
    )

    print("\n최적화 완료!")
    best_config = optimization_results.get('best_config', {})
    if best_config:
        print("\n최적 설정 요약:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

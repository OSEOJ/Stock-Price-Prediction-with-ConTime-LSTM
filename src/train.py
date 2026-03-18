"""
모델 학습, 그리드 서치, 저장 유틸리티 모듈
"""
import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from .contime import build_contime_lstm_model
from .evaluate import (
    backtest_by_ticker, get_risk_free_rate, backtest_buy_and_hold,
    evaluate_model, calculate_combined_score, predict_model,
    find_optimal_threshold,
)
from .plots import (
    plot_training_history, plot_performance_grid, plot_signal_distribution,
    plot_price_predictions
)
from .data.pipeline import prepare_data, clean_numeric_data
from .utils import get_project_root, ensure_directory
from .config import (
    DEFAULT_CONFIG, COMMISSION, MIN_TRADES_RATIO, MAX_TRADES_RATIO,
    THRESHOLD_N_CANDIDATES,
    DEFAULT_WINDOW_SIZE, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS,
    DEFAULT_PATIENCE, DEFAULT_LR_FACTOR, WARMUP_EPOCHS, LR_DECAY_STEPS,
    DEFAULT_DROPOUT, DEFAULT_LEARNING_RATE,
)



def save_model(model, model_path, config=None, encoders=None):
    """PyTorch 모델을 저장합니다."""
    model_path = Path(model_path)
    ensure_directory(model_path.parent)

    if not str(model_path).endswith('.pt'):
        model_path = Path(str(model_path).replace('.keras', '').replace('.h5', '') + '.pt')

    torch.save(model.state_dict(), str(model_path))
    print(f"모델이 {model_path}에 저장되었습니다.")

    model_stem = model_path.stem

    if encoders is not None:
        encoder_path = model_path.parent / f"{model_stem}_encoders.json"
        with open(encoder_path, 'w') as f:
            json.dump(encoders, f, indent=2)
        print(f"인코더 정보가 {encoder_path}에 저장되었습니다.")

    if config is not None:
        config_path = model_path.parent / f"{model_stem}_config.json"
        with open(config_path, 'w') as f:
            json_safe_config = {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v
                              for k, v in config.items()}
            json.dump(json_safe_config, f, indent=2)
        print(f"모델 설정이 {config_path}에 저장되었습니다.")

    return True


def save_results(results, output_path, include_model=False):
    """최적화 결과를 저장합니다."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    pickle_safe_results = {
        'grid_results': [],
        'best_config': results.get('best_config', {})
    }

    results_list = results.get('results', [])
    if not results_list and 'best_result' in results:
        results_list = [results['best_result']]

    for result in results_list:
        result_copy = result.copy()
        if not include_model and 'model' in result_copy:
            del result_copy['model']
        pickle_safe_results['grid_results'].append(result_copy)

    if 'test_backtest' in results:
        pickle_safe_results['test_backtest'] = results['test_backtest']

    with open(output_path, 'wb') as f:
        pickle.dump(pickle_safe_results, f)
    print(f"최적화 결과가 {output_path}에 저장되었습니다.")

    return True


def save_metadata(metadata, output_path):
    """메타데이터를 JSON 형식으로 저장합니다."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    json_safe_metadata = {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v
                       for k, v in metadata.items()}

    with open(output_path, 'w') as f:
        json.dump(json_safe_metadata, f, indent=2)
    print(f"메타데이터가 {output_path}에 저장되었습니다.")

    return True


# ─────────────────────────────────────────────
# 학습
# ─────────────────────────────────────────────

def get_device():
    """사용 가능한 디바이스를 반환합니다."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloader(x, time_diffs, y, y_dt, batch_size, shuffle):
    """numpy 배열로부터 DataLoader를 생성합니다."""
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(time_diffs, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(y_dt, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_one_epoch(model, train_loader, val_loader, optimizer, value_weight, derivative_weight, device):
    """train + val 1 epoch 실행, (train_loss, val_loss) 반환"""
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        x_b, td_b, y_b, ydt_b = [b.to(device) for b in batch]
        optimizer.zero_grad()
        val_out, der_out = model(x_b, td_b)
        loss = (value_weight * F.mse_loss(val_out[:, -1, :], y_b) +
                derivative_weight * F.mse_loss(der_out[:, -1, :], ydt_b))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x_b, td_b, y_b, ydt_b = [b.to(device) for b in batch]
            val_out, der_out = model(x_b, td_b)
            loss = (value_weight * F.mse_loss(val_out[:, -1, :], y_b) +
                    derivative_weight * F.mse_loss(der_out[:, -1, :], ydt_b))
            val_loss += loss.item()

    return train_loss / len(train_loader), val_loss / len(val_loader)


def train_model(model, train_loader, val_loader, config, value_weight, derivative_weight, device):
    """
    PyTorch 학습 루프
    - Warmup: WARMUP_EPOCHS 동안 0 → base_lr 선형 증가
    - LR_DECAY_STEPS회 감쇠, 감쇠마다 전역 best state로 재시작
    - 각 구간에서 EarlyStopping 적용
    반환: history dict {'loss': [...], 'val_loss': [...], 'lr': [...]}
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)

    total_epochs = config['epochs']
    remaining = max(total_epochs - WARMUP_EPOCHS, 1)
    phase_epochs = max(remaining // (LR_DECAY_STEPS + 1), 1)

    global_best_loss = float('inf')
    global_best_state = None
    history = {'loss': [], 'val_loss': [], 'lr': []}

    # ── Warmup 구간 ──────────────────────────────
    print(f"[Warmup] {WARMUP_EPOCHS} epochs")
    for epoch in range(WARMUP_EPOCHS):
        warmup_lr = DEFAULT_LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
        for pg in optimizer.param_groups:
            pg['lr'] = warmup_lr

        train_loss, val_loss = _run_one_epoch(
            model, train_loader, val_loader, optimizer,
            value_weight, derivative_weight, device
        )
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(warmup_lr)

        if val_loss < global_best_loss:
            global_best_loss = val_loss
            global_best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  warmup {epoch+1}/{WARMUP_EPOCHS} | loss {train_loss:.4f} | val {val_loss:.4f} | lr {warmup_lr:.2e}", flush=True)

    # ── 감쇠 구간 (LR_DECAY_STEPS + 1 phases) ──
    current_lr = DEFAULT_LEARNING_RATE
    for phase in range(LR_DECAY_STEPS + 1):
        if phase > 0:
            current_lr *= DEFAULT_LR_FACTOR
            model.load_state_dict(global_best_state)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
            print(f"\n[Phase {phase}/{LR_DECAY_STEPS}] best state 복원 후 lr={current_lr:.2e} 으로 재시작")
        else:
            print(f"\n[Phase {phase+1}/{LR_DECAY_STEPS+1}] lr={current_lr:.2e}")

        patience_counter = 0
        pbar = tqdm(range(phase_epochs), desc=f"Phase {phase+1}", unit="epoch", leave=True)
        for epoch in pbar:
            train_loss, val_loss = _run_one_epoch(
                model, train_loader, val_loader, optimizer,
                value_weight, derivative_weight, device
            )
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(current_lr)

            pbar.set_description(f"loss {train_loss:.4f} | val {val_loss:.4f}")

            if val_loss < global_best_loss:
                global_best_loss = val_loss
                global_best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    print(f"  EarlyStopping: phase {phase+1} epoch {epoch+1}")
                    break

    model.load_state_dict(global_best_state)
    print(f"\n학습 완료! 최적 val_loss: {global_best_loss:.4f}")
    return history


# ─────────────────────────────────────────────
# 그리드 서치
# ─────────────────────────────────────────────

def evaluate_config(config, data_dict, risk_free_rate, selection_method='combined_score'):
    """특정 설정에 대한 모델을 훈련하고 평가합니다."""
    try:
        device = get_device()

        x_train = clean_numeric_data(data_dict['x_train'], verbose=False)
        y_train = data_dict['y_train']
        y_train_dt = data_dict.get('y_train_dt', None)

        x_val = clean_numeric_data(data_dict['x_val'], verbose=False)
        y_val = data_dict['y_val']
        y_val_dt = data_dict.get('y_val_dt', None)
        ticker_val = data_dict['ticker_val']  # 백테스트용

        if y_train_dt is None:
            y_train_dt = np.zeros_like(y_train)
        if y_val_dt is None:
            y_val_dt = np.zeros_like(y_val)

        train_loader = make_dataloader(
            x_train, data_dict['time_diffs_train'], y_train, y_train_dt,
            batch_size=config['batch_size'], shuffle=True
        )
        val_loader = make_dataloader(
            x_val, data_dict['time_diffs_val'], y_val, y_val_dt,
            batch_size=config['batch_size'], shuffle=False
        )

        model = build_contime_lstm_model(
            seq_len=x_train.shape[1],
            num_features=x_train.shape[2],
            hidden_dim=config['hidden_dim'],
            dropout_rate=config['dropout_rate'],
            dt=config['dt'],
            ode_steps=config['ode_steps'],
        ).to(device)

        history = train_model(
            model, train_loader, val_loader, config,
            value_weight=config['value_weight'],
            derivative_weight=config['derivative_weight'],
            device=device
        )

        metrics = evaluate_model(
            model, x_val, y_val, y_val_dt,
            time_diffs_test=data_dict['time_diffs_val'],
            device=device,
            verbose=False
        )

        val_preds, _ = predict_model(
            model, x_val,
            data_dict['time_diffs_val'], device=device
        )

        y_pred_val = val_preds[:, -1, 0]
        y_val_flat = np.asarray(y_val).flatten()
        ticker_val_flat = np.asarray(ticker_val).flatten()

        min_len = min(len(y_pred_val), len(y_val_flat), len(ticker_val_flat))
        y_pred_val = y_pred_val[:min_len]
        y_val_flat = y_val_flat[:min_len]
        ticker_val_flat = ticker_val_flat[:min_len]

        num_tickers = len(np.unique(ticker_val_flat))
        trading_days = len(y_val_flat) // num_tickers
        total_opportunities = trading_days * num_tickers
        min_expected_trades = max(5, int(total_opportunities * MIN_TRADES_RATIO))

        use_combined_score = (selection_method == 'combined_score')
        best_threshold, best_backtest, all_thresholds = find_optimal_threshold(
            y_pred_val, y_val_flat, ticker_val_flat, risk_free_rate,
            min_expected_trades, use_combined_score, total_opportunities
        )

        if selection_method == 'combined_score':
            combined_metric = calculate_combined_score(
                best_backtest,
                total_opportunities=total_opportunities,
                min_trades_ratio=MIN_TRADES_RATIO,
                max_trades_ratio=MAX_TRADES_RATIO
            )
            metrics.update({
                'combined_score': combined_metric,
                'best_threshold': best_threshold,
                'total_return': best_backtest['portfolio']['total_return'],
                'sharpe_ratio': best_backtest['portfolio']['sharpe_ratio'],
                'max_drawdown': best_backtest['portfolio']['max_drawdown'],
                'trade_count': len(best_backtest['portfolio'].get('trades', [])),
                'win_rate': best_backtest['portfolio'].get('win_rate', 0),
                'avg_ticker_sharpe': best_backtest['avg_ticker_sharpe'],
                'dtw': best_backtest['portfolio'].get('dtw', 0),
                'tdi': best_backtest['portfolio'].get('tdi', 0)
            })
        else:
            metrics.update({
                'best_threshold': best_threshold,
                'total_return': best_backtest['portfolio']['total_return'],
                'sharpe_ratio': best_backtest['portfolio']['sharpe_ratio'],
                'max_drawdown': best_backtest['portfolio']['max_drawdown'],
                'trade_count': len(best_backtest['portfolio'].get('trades', [])),
                'win_rate': best_backtest['portfolio'].get('win_rate', 0),
                'avg_ticker_sharpe': best_backtest['avg_ticker_sharpe'],
                'dtw': best_backtest['portfolio'].get('dtw', 0),
                'tdi': best_backtest['portfolio'].get('tdi', 0)
            })

        print(f"임계값: {best_threshold:.4f}, 수익률: {best_backtest['portfolio']['total_return']:.4f}")
        print(f"거래: {len(best_backtest['portfolio'].get('trades', []))}/{total_opportunities} "
              f"({len(best_backtest['portfolio'].get('trades', [])) / total_opportunities:.1%})")
        print(f"DTW: {best_backtest['portfolio'].get('dtw', 0):.4f}, TDI: {best_backtest['portfolio'].get('tdi', 0):.4f}")

        if selection_method == 'combined_score':
            print(f"복합 점수: {combined_metric:.4f}")

        return {
            'config': config,
            'metrics': metrics,
            'model': model,
            'history': history,
            'best_threshold': best_threshold,
            'ticker_metrics': best_backtest['by_ticker'],
            'total_opportunities': total_opportunities,
            'min_expected_trades': min_expected_trades,
            'all_thresholds': all_thresholds
        }

    except Exception as e:
        import traceback
        print(f"모델 평가 실패: {str(e)}")
        traceback.print_exc()
        return None




def run_optimization_pipeline(data_dict, metric='combined_score',
                              output_path=None, save=True, model_output=None,
                              run_visualizations=False):
    """연속 시간 모델 최적화 파이프라인"""
    print("===== ConTime-LSTM 학습 시작 =====")

    device = get_device()
    print(f"디바이스: {device}")

    start_date = data_dict.get('start_date')
    end_date = data_dict.get('end_date')
    risk_free_rate = get_risk_free_rate(start_date, end_date)
    print(f"무위험 수익률: {risk_free_rate:.6f}")

    plots_dir = Path(get_project_root()) / "output" / "plots"
    if run_visualizations:
        plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"시각화 결과는 {plots_dir}에 저장됩니다.")

    required_keys = ['x_train', 'y_train', 'ticker_train', 'time_diffs_train',
                     'x_val', 'y_val', 'ticker_val', 'time_diffs_val']
    if not all(key in data_dict for key in required_keys):
        print("필요한 데이터 키가 없습니다. 데이터를 준비합니다...")
        data_dict, _, _ = prepare_data(data_dict.get('data'), window_size=DEFAULT_WINDOW_SIZE)

    config = {**DEFAULT_CONFIG}
    config['derivative_weight'] = 1.0 - config['value_weight']
    config['dropout_rate'] = DEFAULT_DROPOUT
    config['batch_size']   = DEFAULT_BATCH_SIZE
    config['epochs']       = DEFAULT_EPOCHS
    config['patience']     = DEFAULT_PATIENCE

    best_result = evaluate_config(config, data_dict, risk_free_rate, selection_method=metric)

    if best_result is None:
        print("모델 평가가 실패했습니다.")
        return {'error': '모델 평가 실패', 'best_config': None, 'results': []}

    best_config    = best_result['config']
    best_threshold = best_result['best_threshold']
    best_model     = best_result['model']

    print(f"\n학습 완료!")
    print(f"   수익률: {best_result['metrics'].get('total_return', 0):.4f}")
    print(f"   샤프 비율: {best_result['metrics'].get('sharpe_ratio', 0):.4f}")
    print(f"   거래 수: {best_result['metrics'].get('trade_count', 0)}")

    ticker_metrics = best_result['ticker_metrics']
    ticker_ids = list(ticker_metrics.keys())
    n_tickers = len(ticker_ids)

    avg_return = np.mean([ticker_metrics[tid]['total_return'] for tid in ticker_ids])
    avg_sharpe = np.mean([ticker_metrics[tid]['sharpe_ratio'] for tid in ticker_ids])
    avg_mdd = np.mean([ticker_metrics[tid]['max_drawdown'] for tid in ticker_ids])
    avg_win_rate = np.mean([ticker_metrics[tid].get('win_rate', 0) for tid in ticker_ids])
    total_trades = sum([len(ticker_metrics[tid].get('trades', [])) for tid in ticker_ids])
    avg_trades = total_trades / n_tickers

    print(f"\n----- 종목별 평균 성능 (티커 수: {n_tickers}) -----")
    print(f"평균 종목 수익률: {avg_return:.4f}")
    print(f"평균 종목 샤프 비율: {avg_sharpe:.4f}")
    print(f"평균 종목 최대 낙폭: {avg_mdd:.4f}")
    print(f"평균 종목 승률: {avg_win_rate:.2%}")
    print(f"평균 종목 거래 횟수: {avg_trades:.1f}\n")

    if run_visualizations:
        try:
            fig1 = plot_training_history(best_result['history'])
            if fig1:
                fig1.savefig(plots_dir / "training_history.png", dpi=300, bbox_inches='tight')
                plt.close(fig1)
                print("  - training_history.png 저장 완료")

            fig2 = plot_performance_grid({0.0025: best_result.get('all_thresholds', {})})
            if fig2:
                fig2.savefig(plots_dir / "performance_grid.png", dpi=300, bbox_inches='tight')
                plt.close(fig2)
                print("  - performance_grid.png 저장 완료")

            x_val_clean = clean_numeric_data(data_dict['x_val'], replace_nan=0.0, replace_inf=0.0, verbose=False)
            time_diffs_val = np.asarray(data_dict['time_diffs_val'], dtype=np.float32)

            val_preds, _ = predict_model(best_model, x_val_clean, time_diffs_val, device=device)
            y_pred_val = val_preds[:, -1, 0]

            fig3 = plot_signal_distribution(y_pred_val, best_threshold)
            if fig3:
                fig3.savefig(plots_dir / "signal_distribution.png", dpi=300, bbox_inches='tight')
                plt.close(fig3)
                print("  - signal_distribution.png 저장 완료")

            if 'x_test' in data_dict and len(data_dict['x_test']) > 0:
                x_test = data_dict['x_test']
                ticker_test = data_dict['ticker_test']
                time_diffs_test = data_dict['time_diffs_test']
                x_test_clean = clean_numeric_data(x_test, replace_nan=0.0, replace_inf=0.0, verbose=False)

                test_preds, _ = predict_model(
                    best_model, x_test_clean,
                    np.asarray(time_diffs_test, dtype=np.float32),
                    device=device
                )
                y_pred_viz = test_preds[:, -1, 0]

                viz_backtest = backtest_by_ticker(
                    predictions=y_pred_viz,
                    actual_returns=data_dict['y_test'].flatten(),
                    ticker_ids=np.asarray(ticker_test).flatten(),
                    threshold=best_threshold,
                    commission=COMMISSION,
                    risk_free_rate=risk_free_rate
                )

                test_viz_dict = {
                    'x_test': x_test, 'ticker_test': ticker_test,
                    'time_diffs_test': time_diffs_test,
                    'data': data_dict['data'], 'backtest_result': viz_backtest
                }

                fig5 = plot_price_predictions(best_model, test_viz_dict, best_threshold, None, device, x_test_clean)
                if fig5:
                    fig5.savefig(plots_dir / "price_predictions.png", dpi=300, bbox_inches='tight')
                    plt.close(fig5)
                    print("  - price_predictions.png 저장 완료")

        except Exception as e:
            import traceback
            print(f"시각화 중 오류 발생: {e}")
            traceback.print_exc()

    test_backtest = None
    buy_hold_backtest = None
    if all(key in data_dict for key in ['x_test', 'y_test', 'ticker_test', 'time_diffs_test']):
        print("\n===== 테스트 세트 성능 평가 =====")
        try:
            x_test = data_dict['x_test']
            y_test = data_dict['y_test']
            ticker_test = data_dict['ticker_test']
            time_diffs_test = data_dict['time_diffs_test']

            x_test_clean = clean_numeric_data(x_test, replace_nan=0.0, replace_inf=0.0, verbose=False)

            test_preds, _ = predict_model(
                best_model, x_test_clean,
                np.asarray(time_diffs_test, dtype=np.float32),
                device=device
            )
            y_pred_test = test_preds[:, -1, 0]

            buy_hold_backtest = backtest_buy_and_hold(
                actual_returns=y_test.flatten(),
                ticker_ids=np.asarray(ticker_test).flatten(),
                commission=COMMISSION,
                risk_free_rate=risk_free_rate
            )

            test_backtest = backtest_by_ticker(
                predictions=y_pred_test,
                actual_returns=y_test.flatten(),
                ticker_ids=np.asarray(ticker_test).flatten(),
                threshold=best_threshold,
                commission=COMMISSION,
                risk_free_rate=risk_free_rate
            )

            ticker_returns = [info['total_return'] for _, info in test_backtest['by_ticker'].items()]
            ticker_sharpes = [info['sharpe_ratio'] for _, info in test_backtest['by_ticker'].items()]
            avg_ticker_return = np.mean(ticker_returns)
            avg_ticker_sharpe = np.mean(ticker_sharpes)

            bh_ticker_returns = [info['total_return'] for _, info in buy_hold_backtest['by_ticker'].items()]
            bh_ticker_sharpes = [info['sharpe_ratio'] for _, info in buy_hold_backtest['by_ticker'].items()]
            bh_avg_ticker_return = np.mean(bh_ticker_returns)
            bh_avg_ticker_sharpe = np.mean(bh_ticker_sharpes)

            return_improvement = test_backtest['portfolio']['total_return'] - buy_hold_backtest['portfolio']['total_return']

            print(f"\n----- Buy and Hold 전략 성능 -----")
            print(f"Buy & Hold 평균 종목 수익률: {bh_avg_ticker_return:.4f}")
            print(f"Buy & Hold 평균 종목 샤프 비율: {bh_avg_ticker_sharpe:.4f}")

            print(f"\n----- 모델 기반 전략 성능 -----")
            print(f"모델 평균 종목 수익률: {avg_ticker_return:.4f}")
            print(f"모델 평균 종목 샤프 비율: {avg_ticker_sharpe:.4f}")
            print(f"모델 거래 수: {len(test_backtest['portfolio'].get('trades', []))}")
            print(f"모델 거래 승률: {test_backtest['portfolio'].get('trade_win_rate', 0):.2%} "
                  f"({test_backtest['portfolio'].get('winning_trades', 0)}/{test_backtest['portfolio'].get('total_trade_pairs', 0)})")

            test_dtw = test_backtest['portfolio'].get('dtw', 0)
            test_tdi = test_backtest['portfolio'].get('tdi', 0)
            print(f"모델 DTW (Dynamic Time Warping): {test_dtw:.4f}")
            print(f"모델 TDI (Temporal Distortion Index): {test_tdi:.4f}")

            print(f"\n----- 성능 비교 (모델 vs Buy & Hold) -----")
            print(f"수익률 개선: {return_improvement:+.4f} ({return_improvement/abs(buy_hold_backtest['portfolio']['total_return'])*100:+.1f}%)")

        except Exception as e:
            import traceback
            print(f"테스트 세트 평가 중 오류 발생: {e}")
            traceback.print_exc()

    optimization_results = {
        'best_config': best_config,
        'best_result': best_result,
        'solver': 'rk4'
    }

    if save:
        if best_model:
            results_dir = Path(get_project_root()) / "output" / "checkpoints"
            results_dir.mkdir(parents=True, exist_ok=True)

            if model_output is None:
                model_output = results_dir / "best_contime_model.pt"
            else:
                model_output = Path(model_output)
                if not model_output.is_absolute():
                    model_output = results_dir / model_output.name

            save_model(model=best_model, model_path=model_output, config=best_config)

            threshold_info = {
                'best_threshold': best_threshold,
                'config': best_config,
                'avg_ticker_sharpe': best_result['metrics'].get('avg_ticker_sharpe', 0),
                'portfolio_sharpe': best_result['metrics']['sharpe_ratio'],
                'total_return': best_result['metrics']['total_return'],
                'avg_ticker_return': float(avg_return),
                'avg_ticker_win_rate': float(avg_win_rate),
                'avg_ticker_mdd': float(avg_mdd),
                'trade_count': best_result['metrics']['trade_count'],
                'total_opportunities': best_result['total_opportunities'],
                'trade_ratio': float(best_result['metrics']['trade_count'] / best_result['total_opportunities']),
                'min_expected_trades': best_result['min_expected_trades']
            }

            if test_backtest:
                threshold_info['test_metrics'] = {
                    'total_return': test_backtest['portfolio']['total_return'],
                    'sharpe_ratio': test_backtest['portfolio']['sharpe_ratio'],
                    'max_drawdown': test_backtest['portfolio']['max_drawdown'],
                    'trade_count': len(test_backtest['portfolio'].get('trades', [])),
                    'dtw': test_backtest['portfolio'].get('dtw', 0),
                    'tdi': test_backtest['portfolio'].get('tdi', 0)
                }

            meta_path = Path(get_project_root()) / "output" / "checkpoints" / f"{Path(str(model_output)).stem}_meta.json"
            save_metadata(threshold_info, meta_path)

        if output_path:
            output_path = Path(output_path)
            if not output_path.is_absolute():
                output_path = Path(get_project_root()) / "output" / "checkpoints" / output_path
            save_results(optimization_results, output_path)

    if run_visualizations:
        print(f"\n시각화 파일들이 {plots_dir}에 저장되었습니다:")
        saved_plots = list(plots_dir.glob("*.png"))
        if saved_plots:
            for plot_file in saved_plots:
                print(f"  - {plot_file.name}")
        else:
            print("  - 저장된 시각화 파일이 없습니다.")

    return optimization_results

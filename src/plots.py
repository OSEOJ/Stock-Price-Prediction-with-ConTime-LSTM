"""
시각화 함수 모듈
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import traceback
from .evaluate import predict_model

def clean_for_visualization(X):
    """시각화를 위한 간단한 데이터 정리 함수"""
    if X is None:
        return X

    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X

def plot_training_history(history):
    """학습 과정의 손실과 학습률을 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    history_dict = history.history if hasattr(history, 'history') else history

    ax1.plot(history_dict['loss'], label='Train Loss')
    ax1.plot(history_dict['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history_dict['lr'], label='Learning Rate')
    ax2.set_title('Learning Rate')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    return fig

def plot_performance_grid(grid_results):
    """임계값별 성능 지표를 그리드로 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))

    for comm, res in grid_results.items():
        ths = list(res.keys())
        rets = [res[t]['total_return'] for t in ths]
        axes[0, 0].plot(ths, rets, label=f'Comm {comm*100:.2f}%')
    axes[0, 0].set_title('Total Return by Threshold')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Total Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    for comm, res in grid_results.items():
        ths = list(res.keys())
        sps = [res[t]['sharpe_ratio'] for t in ths]
        axes[0, 1].plot(ths, sps, label=f'Comm {comm*100:.2f}%')
    axes[0, 1].set_title('Sharpe Ratio by Threshold')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    for comm, res in grid_results.items():
        ths = list(res.keys())
        tcs = [len(res[t].get('trades', [])) for t in ths]
        axes[1, 0].plot(ths, tcs, label=f'Comm {comm*100:.2f}%')
    axes[1, 0].set_title('Number of Trades')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Trades')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    for comm, res in grid_results.items():
        ths = list(res.keys())
        mdds = [res[t]['max_drawdown'] for t in ths]
        axes[1, 1].plot(ths, mdds, label=f'Comm {comm*100:.2f}%')
    axes[1, 1].set_title('Max Drawdown by Threshold')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Max Drawdown')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    return fig

def plot_signal_distribution(y_pred, best_threshold):
    """예측 신호의 분포를 시각화"""
    fig, ax = plt.subplots(figsize=(12, 6))

    buy_signals = y_pred > best_threshold
    sell_signals = y_pred < -best_threshold
    hold_signals = (y_pred >= -best_threshold) & (y_pred <= best_threshold)

    ax.hist([y_pred[buy_signals], y_pred[sell_signals], y_pred[hold_signals]],
             bins=50, label=['Buy', 'Sell', 'Hold'], alpha=0.7)
    ax.axvline(x=best_threshold, color='r', linestyle='--', label=f'Buy Threshold ({best_threshold:.4f})')
    ax.axvline(x=-best_threshold, color='g', linestyle='--', label=f'Sell Threshold (-{best_threshold:.4f})')
    ax.set_xlabel('Predicted Returns')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Trading Signals')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    return fig

def plot_price_predictions(model, data_dict, best_threshold, ticker_encoder, device, x_test_clean=None):
    """
    모델의 예측 변동률을 기반으로 예측 종가를 계산하고 실제 종가와 함께 시각화
    매수/매도 시점과 주식 보유 기간을 표시
    """
    try:
        backtest_result = data_dict.get('backtest_result')

        x_test = data_dict['x_test']
        ticker_test = data_dict['ticker_test']
        data = data_dict['data']
        time_diffs_test = data_dict.get('time_diffs_test')

        if len(x_test) == 0:
            x_test = data_dict['x_val']
            ticker_test = data_dict['ticker_val']
            time_diffs_test = data_dict.get('time_diffs_val')

        if len(x_test) == 0:
            return None

        if x_test_clean is not None:
            x_test_processed = x_test_clean
        else:
            x_test_processed = clean_for_visualization(x_test)

        ticker_test_clean = np.asarray(ticker_test, dtype=np.int32)

        sector_test = data_dict.get('sector_test')
        industry_test = data_dict.get('industry_test')

        n_samples = x_test_processed.shape[0]

        if len(ticker_test_clean) != n_samples:
            if len(ticker_test_clean) > n_samples:
                ticker_test_clean = ticker_test_clean[:n_samples]
            else:
                last_ticker = ticker_test_clean[-1] if len(ticker_test_clean) > 0 else 0
                ticker_test_clean = np.pad(ticker_test_clean, (0, n_samples - len(ticker_test_clean)),
                                         mode='constant', constant_values=last_ticker)

        if (sector_test is None or industry_test is None) and x_test_processed.shape[2] > 60:
            sector_feature_idx = 59
            industry_feature_idx = 60
            sector_test = x_test_processed[:, -1, sector_feature_idx].astype(np.int32)
            industry_test = x_test_processed[:, -1, industry_feature_idx].astype(np.int32)

        if sector_test is None or industry_test is None:
            sector_test = np.zeros(n_samples, dtype=np.int32)
            industry_test = np.zeros(n_samples, dtype=np.int32)
        else:
            sector_test = np.asarray(sector_test, dtype=np.int32)
            industry_test = np.asarray(industry_test, dtype=np.int32)

            if len(sector_test) != n_samples:
                if len(sector_test) > n_samples:
                    sector_test = sector_test[:n_samples]
                else:
                    last_sector = sector_test[-1] if len(sector_test) > 0 else 0
                    sector_test = np.pad(sector_test, (0, n_samples - len(sector_test)),
                                       mode='constant', constant_values=last_sector)

            if len(industry_test) != n_samples:
                if len(industry_test) > n_samples:
                    industry_test = industry_test[:n_samples]
                else:
                    last_industry = industry_test[-1] if len(industry_test) > 0 else 0
                    industry_test = np.pad(industry_test, (0, n_samples - len(industry_test)),
                                         mode='constant', constant_values=last_industry)

        if time_diffs_test is None or len(time_diffs_test) == 0:
            time_diffs_test = np.ones((n_samples, x_test_processed.shape[1]), dtype=np.float32)
        else:
            time_diffs_test = np.asarray(time_diffs_test, dtype=np.float32)

            if time_diffs_test.shape[0] != n_samples:
                if time_diffs_test.shape[0] > n_samples:
                    time_diffs_test = time_diffs_test[:n_samples]
                else:
                    padding_shape = (n_samples - time_diffs_test.shape[0], time_diffs_test.shape[1])
                    padding = np.ones(padding_shape, dtype=np.float32)
                    time_diffs_test = np.vstack([time_diffs_test, padding])

        # 예측 수행
        raw_preds, _ = predict_model(
            model,
            x_test_processed,
            ticker_test_clean,
            sector_test,
            industry_test,
            time_diffs_test,
            device=device,
        )
        test_pred_values = raw_preds[:, -1, 0]

        ticker_test_flat = ticker_test.flatten()
        if len(test_pred_values) != len(ticker_test_flat):
            seq_len = x_test_processed.shape[1]

            if len(test_pred_values) == len(ticker_test_flat) * seq_len:
                test_pred_values = test_pred_values.reshape(-1, seq_len)[:, -1]
            else:
                factor = int(len(test_pred_values) / len(ticker_test_flat))
                if factor > 1:
                    test_pred_values = test_pred_values[factor-1::factor]

        unique_tickers = np.unique(ticker_test)

        fig, axes = plt.subplots(len(unique_tickers), 1, figsize=(15, len(unique_tickers) * 4))
        if len(unique_tickers) == 1:
            axes = [axes]

        total_buy_signals = 0
        total_sell_signals = 0

        backtest_trades = {}
        if backtest_result:
            for ticker_id, ticker_info in backtest_result['by_ticker'].items():
                ticker_trades = ticker_info.get('trades', [])
                backtest_trades[int(ticker_id)] = ticker_trades

        for i, ticker_id in enumerate(unique_tickers):
            ticker_mask = ticker_test_flat == ticker_id
            ticker_indices = np.where(ticker_mask)[0]

            ticker_name = ticker_encoder.inverse_transform([int(ticker_id)])[0]
            ticker_preds = test_pred_values[ticker_mask]

            ticker_data = data[data['ticker'] == ticker_name].copy()
            ticker_data = ticker_data.sort_index()

            actual_prices = ticker_data['Close'].values[-len(ticker_preds)-1:]
            dates = ticker_data.index[-len(ticker_preds)-1:]

            predicted_prices = []
            last_price = actual_prices[0]

            for j, pred in enumerate(ticker_preds):
                predicted_return = np.exp(pred) - 1
                predicted_price = last_price * (1 + predicted_return)
                predicted_prices.append(predicted_price)
                last_price = actual_prices[j+1]

            if int(ticker_id) in backtest_trades:
                trades = backtest_trades[int(ticker_id)]
                buy_signals = []
                sell_signals = []
                holding_periods = []

                for trade in trades:
                    global_day = trade['day']

                    if global_day in ticker_indices:
                        local_day = np.where(ticker_indices == global_day)[0]

                        if len(local_day) > 0:
                            local_day = local_day[0]

                            if 0 <= local_day < len(ticker_preds):
                                if trade['action'] == 'BUY':
                                    buy_signals.append(local_day)
                                elif trade['action'] == 'SELL':
                                    sell_signals.append(local_day)

                buy_signals.sort()
                sell_signals.sort()

                for buy_idx in buy_signals:
                    corresponding_sell = None
                    for sell_idx in sell_signals:
                        if sell_idx > buy_idx:
                            corresponding_sell = sell_idx
                            break

                    if corresponding_sell is not None:
                        holding_periods.append((buy_idx, corresponding_sell))
                    else:
                        holding_periods.append((buy_idx, len(ticker_preds) - 1))

                total_buy_signals += len(buy_signals)
                total_sell_signals += len(sell_signals)

            else:
                current_position = 0
                buy_signals = []
                sell_signals = []
                holding_periods = []
                holding_start = None

                for j, pred in enumerate(ticker_preds):
                    if pred > best_threshold and current_position == 0:
                        current_position = 1
                        buy_signals.append(j)
                        holding_start = j
                    elif pred < -best_threshold and current_position == 1:
                        current_position = 0
                        sell_signals.append(j)
                        if holding_start is not None:
                            holding_periods.append((holding_start, j))
                            holding_start = None

                if holding_start is not None and current_position == 1:
                    holding_periods.append((holding_start, len(ticker_preds) - 1))

                total_buy_signals += len(buy_signals)
                total_sell_signals += len(sell_signals)

            axes[i].plot(dates[1:], actual_prices[1:], 'b-', label='Actual Price', linewidth=2)
            axes[i].plot(dates[1:], predicted_prices, 'r--', label='Predicted Price', linewidth=1.5, alpha=0.8)

            for period_idx, (start_idx, end_idx) in enumerate(holding_periods):
                start_idx = max(0, min(start_idx, len(dates[1:]) - 1))
                end_idx = max(0, min(end_idx, len(dates[1:]) - 1))

                if start_idx < len(dates[1:]) and end_idx < len(dates[1:]) and start_idx <= end_idx:
                    axes[i].axvspan(dates[1:][start_idx], dates[1:][end_idx],
                                alpha=0.2, color='lightgreen',
                                label='Holding Period' if period_idx == 0 else "")

            if buy_signals:
                buy_indices = np.array(buy_signals)
                valid_buy_indices = buy_indices[buy_indices < len(dates[1:])]

                if len(valid_buy_indices) > 0:
                    axes[i].scatter(dates[1:][valid_buy_indices], actual_prices[1:][valid_buy_indices],
                              marker='^', color='darkgreen', s=200, label='Buy Signal',
                              edgecolors='white', linewidth=2, zorder=6, alpha=0.9)

            if sell_signals:
                sell_indices = np.array(sell_signals)
                valid_sell_indices = sell_indices[sell_indices < len(dates[1:])]

                if len(valid_sell_indices) > 0:
                    axes[i].scatter(dates[1:][valid_sell_indices], actual_prices[1:][valid_sell_indices],
                              marker='v', color='darkred', s=200, label='Sell Signal',
                              edgecolors='white', linewidth=2, zorder=6, alpha=0.9)

            axes[i].set_title(f'{ticker_name} - Price Prediction with Buy/Sell Signals', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Price ($)', fontsize=10)
            axes[i].legend(loc='upper left', fontsize=9)
            axes[i].grid(True, alpha=0.3)

            axes[i].set_xticks([])
            axes[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            if i == len(unique_tickers) - 1:
                axes[i].set_xlabel('Date', fontsize=10)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Price prediction visualization failed: {e}")
        return None


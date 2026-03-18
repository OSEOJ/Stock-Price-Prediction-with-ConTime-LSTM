"""
모델 평가 및 백테스트 모듈
"""
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from torch.utils.data import DataLoader, TensorDataset
from fastdtw import fastdtw
from .config import MIN_TRADES_RATIO, MAX_TRADES_RATIO, COMMISSION as DEFAULT_COMMISSION, THRESHOLD_N_CANDIDATES


# ─────────────────────────────────────────────
# 단독 평가 지표
# ─────────────────────────────────────────────

def direction_accuracy(predictions, actual_returns):
    """예측 방향(상승/하락) 정확도"""
    preds   = np.asarray(predictions).flatten()
    actuals = np.asarray(actual_returns).flatten()
    return float(np.mean(np.sign(preds) == np.sign(actuals)))


def safe_auc(actual_returns, predictions):
    """
    AUC-ROC 점수 계산 (예외 안전).
    actual_returns > 0 이면 상승(1), 아니면 하락(0)으로 이진 변환.
    클래스가 하나뿐이거나 오류 시 nan 반환.
    """
    try:
        from sklearn.metrics import roc_auc_score
        y_bin = (np.asarray(actual_returns).flatten() > 0).astype(int)
        if len(np.unique(y_bin)) < 2:
            return float('nan')
        return float(roc_auc_score(y_bin, np.asarray(predictions).flatten()))
    except Exception:
        return float('nan')


# ─────────────────────────────────────────────
# 모델 추론 / 평가
# ─────────────────────────────────────────────

def predict_model(model, x, time_diffs, device=None, batch_size=64):
    """
    PyTorch 모델로 배치 추론을 수행합니다.
    반환: (value_preds, deriv_preds) 각각 numpy array (n, seq, 1)
    """
    if device is None:
        device = next(model.parameters()).device

    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(time_diffs, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_value, all_deriv = [], []

    with torch.no_grad():
        for batch in loader:
            x_b, td_b = [b.to(device) for b in batch]
            val_out, der_out = model(x_b, td_b)
            all_value.append(val_out.cpu().numpy())
            all_deriv.append(der_out.cpu().numpy())

    return np.concatenate(all_value, axis=0), np.concatenate(all_deriv, axis=0)


def evaluate_model(model, x_test, y_test, y_test_dt=None, time_diffs_test=None,
                   device=None, verbose=True):
    """
    모델을 평가하고 성능 지표를 반환합니다.
    """
    try:
        time_diffs = time_diffs_test if time_diffs_test is not None else np.ones((len(x_test), x_test.shape[1]))

        value_preds, deriv_preds = predict_model(
            model, x_test, time_diffs,
            device=device
        )

        # 마지막 타임스텝 추출
        y_pred = value_preds[:, -1, 0]
        y_test_flat = np.array(y_test).flatten()

        min_len = min(len(y_pred), len(y_test_flat))
        y_pred = y_pred[:min_len]
        y_test_flat = y_test_flat[:min_len]

        # 기본 지표
        mse = np.mean((y_pred - y_test_flat) ** 2)
        mae = np.mean(np.abs(y_pred - y_test_flat))
        rmse = np.sqrt(mse)
        correlation = np.corrcoef(y_pred, y_test_flat)[0, 1] if len(y_pred) > 1 else 0.0
        ss_res = np.sum((y_test_flat - y_pred) ** 2)
        ss_tot = np.sum((y_test_flat - np.mean(y_test_flat)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        direction_accuracy = np.mean(np.sign(y_pred) == np.sign(y_test_flat))

        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'correlation': float(correlation),
            'r2_score': float(r2_score),
            'direction_accuracy': float(direction_accuracy)
        }

        # 도함수 평가
        if y_test_dt is not None:
            y_pred_dt = deriv_preds[:, -1, 0]
            y_test_dt_flat = np.array(y_test_dt).flatten()

            min_len_dt = min(len(y_pred_dt), len(y_test_dt_flat))
            y_pred_dt = y_pred_dt[:min_len_dt]
            y_test_dt_flat = y_test_dt_flat[:min_len_dt]

            dt_mse = np.mean((y_pred_dt - y_test_dt_flat) ** 2)
            dt_mae = np.mean(np.abs(y_pred_dt - y_test_dt_flat))
            dt_correlation = np.corrcoef(y_pred_dt, y_test_dt_flat)[0, 1] if len(y_pred_dt) > 1 else 0.0

            metrics.update({
                'dt_mse': float(dt_mse),
                'dt_mae': float(dt_mae),
                'dt_correlation': float(dt_correlation)
            })

        if verbose:
            print(f"평가 완료 - MSE: {mse:.6f}, MAE: {mae:.6f}, 상관계수: {correlation:.4f}")
            if 'dt_mse' in metrics:
                print(f"도함수 평가 - MSE: {metrics['dt_mse']:.6f}, MAE: {metrics['dt_mae']:.6f}")

        return metrics

    except Exception as e:
        if verbose:
            print(f"모델 평가 중 오류: {e}")
        return {
            'mse': float('inf'),
            'mae': float('inf'),
            'rmse': float('inf'),
            'correlation': 0.0,
            'r2_score': 0.0,
            'direction_accuracy': 0.0
        }


def calculate_dtw(predictions, actual_returns):
    """
    두 시계열 간의 Dynamic Time Warping 거리를 계산합니다.
    """
    try:
        def custom_euclidean(u, v):
            return abs(float(u) - float(v))

        predictions = [float(x) for x in np.array(predictions).flatten()]
        actual_returns = [float(x) for x in np.array(actual_returns).flatten()]

        min_len = min(len(predictions), len(actual_returns))
        predictions = predictions[:min_len]
        actual_returns = actual_returns[:min_len]

        distance, _ = fastdtw(predictions, actual_returns, dist=custom_euclidean)
        return distance

    except Exception as e:
        print(f"DTW 계산 중 오류: {e}")
        import traceback
        print(traceback.format_exc())
        return float('inf')


def calculate_tdi(predictions, actual_returns):
    """
    두 시계열 간의 Temporal Distortion Index를 계산합니다.
    """
    try:
        def custom_euclidean(u, v):
            return abs(float(u) - float(v))

        predictions = [float(x) for x in np.array(predictions).flatten()]
        actual_returns = [float(x) for x in np.array(actual_returns).flatten()]

        min_len = min(len(predictions), len(actual_returns))
        predictions = predictions[:min_len]
        actual_returns = actual_returns[:min_len]

        _, path = fastdtw(predictions, actual_returns, dist=custom_euclidean)
        path = np.array(path)

        P = len(predictions)
        squared_offsets = (path[:, 0] - path[:, 1]) ** 2
        tdi = np.sum(squared_offsets) / (P ** 2)

        return tdi

    except Exception as e:
        print(f"TDI 계산 중 오류: {e}")
        return float('inf')


def calculate_combined_score(backtest_result, total_opportunities=None,
                              min_trades_ratio=MIN_TRADES_RATIO, max_trades_ratio=MAX_TRADES_RATIO):
    """
    거래 횟수와 샤프 비율을 동시에 고려하는 복합 점수 계산
    """
    portfolio = backtest_result.get('portfolio', {})
    trades = len(portfolio.get('trades', []))
    sharpe = portfolio.get('sharpe_ratio', 0)

    dtw = max(portfolio.get('dtw', 1.0), 1e-6)
    tdi = max(portfolio.get('tdi', 1.0), 1e-6)

    min_trades = max(3, int(total_opportunities * min_trades_ratio))
    max_trades = max(min_trades * 2, int(total_opportunities * max_trades_ratio))

    if trades < min_trades:
        trade_weight = (trades / min_trades) ** 0.5
    elif trades > max_trades:
        trade_weight = np.exp(-(trades - max_trades) / max_trades)
    else:
        trade_weight = 1.0

    combined_score = (
        0.7 * sharpe * trade_weight +
        0.15 * (1 / dtw) +
        0.15 * (1 / tdi)
    )

    return combined_score


# ─────────────────────────────────────────────
# 백테스트
# ─────────────────────────────────────────────

def get_risk_free_rate(start_date=None, end_date=None, ticker='^IRX'):
    """
    야후 파이낸스에서 국채 수익률 데이터를 가져옵니다.
    ^TNX: 10년물 미국 국채 수익률
    ^IRX: 13주물 미국 국채 수익률
    """
    try:
        if start_date and isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date and isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        treasury_data = yf.download(ticker, start=start_date, end=end_date)

        if not treasury_data.empty:
            avg_yield_raw = treasury_data['Close'].mean()
            avg_yield = float(avg_yield_raw.iloc[0]) / 100.0
            return avg_yield
    except Exception as e:
        print(f"국채 수익률 데이터 가져오기 오류: {e}")

    default_rate = 0.02
    print(f"국채 수익률을 가져올 수 없습니다. 기본값 {default_rate:.4f}(2%)를 사용합니다.")
    return default_rate


def calculate_max_drawdown(equity_curve):
    """주식 그래프에서 최대 낙폭을 계산"""
    if len(equity_curve) <= 1:
        return 0.0

    equity_curve = np.asarray(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / np.maximum(peak, 1e-10)

    return np.min(drawdown)


def calculate_performance_metrics(portfolio_values, daily_returns, risk_free_rate=0.0):
    """포트폴리오 성능 지표 계산"""
    if len(portfolio_values) <= 1:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'std_dev': 0.0,
            'trades': []
        }

    returns_array = np.array(daily_returns)
    portfolio_values = np.array(portfolio_values)

    non_zero_returns = returns_array[np.abs(returns_array) > 1e-8]

    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1

    n_days = len(portfolio_values) - 1
    if n_days > 0:
        n_years = n_days / 252
        if n_years > 0:
            annualized_return = ((1 + total_return) ** (1 / n_years)) - 1
        else:
            annualized_return = total_return * 252
    else:
        annualized_return = 0.0

    if np.isscalar(risk_free_rate):
        daily_rf_rate = risk_free_rate / 252
    else:
        daily_rf_rate = 0.0

    if len(non_zero_returns) > 1:
        excess_returns = non_zero_returns - daily_rf_rate
        excess_mean = np.mean(excess_returns)
        excess_std = np.std(excess_returns, ddof=1)

        if excess_std > 1e-8:
            trading_frequency = len(non_zero_returns) / len(returns_array)
            annualized_factor = np.sqrt(252 * trading_frequency)
            sharpe_ratio = (excess_mean / excess_std) * annualized_factor
        else:
            sharpe_ratio = 0.0
    else:
        excess_returns = returns_array - daily_rf_rate
        if len(excess_returns) > 1:
            excess_mean = np.mean(excess_returns)
            excess_std = np.std(excess_returns, ddof=1)
            if excess_std > 1e-8:
                sharpe_ratio = (excess_mean / excess_std) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

    max_drawdown = calculate_max_drawdown(portfolio_values)

    if len(non_zero_returns) > 0:
        positive_trading_returns = non_zero_returns[non_zero_returns > 0]
        trading_win_rate = len(positive_trading_returns) / len(non_zero_returns)
    else:
        trading_win_rate = 0.0

    if len(returns_array) > 0:
        positive_all_returns = returns_array[returns_array > 0]
        overall_win_rate = len(positive_all_returns) / len(returns_array)
    else:
        overall_win_rate = 0.0

    std_dev = np.std(returns_array) * np.sqrt(252)

    return {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(trading_win_rate),
        'overall_win_rate': float(overall_win_rate),
        'avg_return': float(np.mean(returns_array)),
        'std_dev': float(std_dev),
        'active_trading_days': len(non_zero_returns),
        'total_days': len(returns_array),
        'trading_frequency': len(non_zero_returns) / len(returns_array) if len(returns_array) > 0 else 0
    }


def calculate_trade_win_rate(trades):
    """실제 거래들의 승률을 계산"""
    if len(trades) < 2:
        return 0.0, 0, 0

    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']

    trade_profits = []
    min_pairs = min(len(buy_trades), len(sell_trades))

    for i in range(min_pairs):
        buy_trade = buy_trades[i]
        sell_trade = sell_trades[i]

        if sell_trade['day'] > buy_trade['day']:
            profit_rate = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
            trade_profits.append(profit_rate)

    if len(trade_profits) == 0:
        return 0.0, 0, 0

    winning_trades = [p for p in trade_profits if p > 0]
    win_rate = len(winning_trades) / len(trade_profits)

    return win_rate, len(winning_trades), len(trade_profits)


def backtest_by_ticker(predictions, actual_returns, ticker_ids, threshold=0.05,
                      commission=DEFAULT_COMMISSION, risk_free_rate=None):
    """개별 종목별 및 전체 포트폴리오 백테스트 함수 - 매수/홀딩 전략"""
    if risk_free_rate is None:
        risk_free_rate = get_risk_free_rate()

    if hasattr(predictions, 'values'):
        predictions = predictions.values
    if hasattr(actual_returns, 'values'):
        actual_returns = actual_returns.values
    if hasattr(ticker_ids, 'values'):
        ticker_ids = ticker_ids.values

    predictions = np.asarray(predictions)
    actual_returns = np.asarray(actual_returns)
    ticker_ids = np.asarray(ticker_ids)

    n_samples = len(predictions)
    unique_tickers = np.unique(ticker_ids)
    n_tickers = len(unique_tickers)

    initial_capital = 1.0
    portfolio_values = [initial_capital]
    portfolio_trades = []
    daily_returns = []

    ticker_results = {ticker_id: {
        'cash': initial_capital / n_tickers,
        'shares': 0,
        'values': [initial_capital / n_tickers],
        'returns': [],
        'trades': [],
        'position': 0,
        'last_price': 1.0,
    } for ticker_id in unique_tickers}

    for t in range(n_samples):
        daily_portfolio_value = 0.0
        previous_portfolio_value = portfolio_values[-1]

        for ticker_id in unique_tickers:
            ticker_mask = ticker_ids == ticker_id
            if not any(ticker_mask[t:t+1]):
                ticker_result = ticker_results[ticker_id]
                current_value = ticker_result['cash'] + ticker_result['shares'] * ticker_result['last_price']
                daily_portfolio_value += current_value
                ticker_result['returns'].append(0.0)
                ticker_result['values'].append(current_value)
                continue

            ticker_pred = predictions[t:t+1][ticker_mask[t:t+1]][0]
            ticker_actual = actual_returns[t:t+1][ticker_mask[t:t+1]][0]

            ticker_result = ticker_results[ticker_id]
            ticker_result['last_price'] *= (1 + ticker_actual)

            if ticker_pred > threshold and ticker_result['position'] == 0:
                new_signal = 1
            elif ticker_pred < -threshold and ticker_result['position'] == 1:
                new_signal = 0
            else:
                new_signal = ticker_result['position']

            current_position = ticker_result['position']

            if new_signal != current_position:
                if current_position == 0 and new_signal == 1:
                    available_cash = ticker_result['cash']
                    shares_to_buy = available_cash / ticker_result['last_price']
                    ticker_result['shares'] = shares_to_buy
                    ticker_result['cash'] = 0
                    ticker_result['position'] = 1
                    trade = {
                        'day': t, 'ticker': ticker_id, 'action': 'BUY',
                        'shares': shares_to_buy, 'price': ticker_result['last_price'],
                        'value': available_cash, 'pred': ticker_pred
                    }
                elif current_position == 1 and new_signal == 0:
                    shares_to_sell = ticker_result['shares']
                    sale_proceeds = shares_to_sell * ticker_result['last_price'] * (1 - commission)
                    ticker_result['cash'] = sale_proceeds
                    ticker_result['shares'] = 0
                    ticker_result['position'] = 0
                    trade = {
                        'day': t, 'ticker': ticker_id, 'action': 'SELL',
                        'shares': shares_to_sell, 'price': ticker_result['last_price'],
                        'value': sale_proceeds, 'pred': ticker_pred
                    }

                portfolio_trades.append(trade)
                ticker_result['trades'].append(trade)

            current_value = ticker_result['cash'] + ticker_result['shares'] * ticker_result['last_price']
            previous_value = ticker_result['values'][-1] if ticker_result['values'] else initial_capital / n_tickers
            ticker_daily_return = (current_value / previous_value) - 1 if previous_value > 0 else 0

            ticker_result['returns'].append(ticker_daily_return)
            ticker_result['values'].append(current_value)
            daily_portfolio_value += current_value

        portfolio_daily_return = (daily_portfolio_value / previous_portfolio_value) - 1 if previous_portfolio_value > 0 else 0
        daily_returns.append(portfolio_daily_return)
        portfolio_values.append(daily_portfolio_value)

    portfolio_metrics = calculate_performance_metrics(portfolio_values, daily_returns, risk_free_rate)
    portfolio_metrics['trades'] = portfolio_trades

    portfolio_trade_win_rate, portfolio_winning_trades, portfolio_total_trades = calculate_trade_win_rate(portfolio_trades)
    portfolio_metrics['trade_win_rate'] = portfolio_trade_win_rate
    portfolio_metrics['winning_trades'] = portfolio_winning_trades
    portfolio_metrics['total_trade_pairs'] = portfolio_total_trades

    for ticker_id in unique_tickers:
        ticker_values = ticker_results[ticker_id]['values']
        ticker_returns = ticker_results[ticker_id]['returns']
        ticker_trades = ticker_results[ticker_id]['trades']

        if len(ticker_returns) > 0:
            ticker_metrics = calculate_performance_metrics(ticker_values, ticker_returns, risk_free_rate)
            ticker_results[ticker_id].update(ticker_metrics)

            ticker_trade_win_rate, ticker_winning_trades, ticker_total_trades = calculate_trade_win_rate(ticker_trades)
            ticker_results[ticker_id]['trade_win_rate'] = ticker_trade_win_rate
            ticker_results[ticker_id]['winning_trades'] = ticker_winning_trades
            ticker_results[ticker_id]['total_trade_pairs'] = ticker_total_trades
        else:
            ticker_results[ticker_id].update({
                'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0,
                'trade_count': 0, 'win_rate': 0, 'trade_win_rate': 0,
                'winning_trades': 0, 'total_trade_pairs': 0,
                'risk_free_rate': risk_free_rate
            })

    try:
        flat_predictions = np.asarray(predictions).flatten()
        flat_actual_returns = np.asarray(actual_returns).flatten()

        mask_pred = ~np.isnan(flat_predictions)
        mask_act = ~np.isnan(flat_actual_returns)
        clean_predictions = flat_predictions[mask_pred]
        clean_actual_returns = flat_actual_returns[mask_act]

        min_len = min(len(clean_predictions), len(clean_actual_returns))
        if min_len > 0:
            clean_predictions = clean_predictions[:min_len]
            clean_actual_returns = clean_actual_returns[:min_len]
            portfolio_metrics['dtw'] = calculate_dtw(clean_predictions, clean_actual_returns)
            portfolio_metrics['tdi'] = calculate_tdi(clean_predictions, clean_actual_returns)
        else:
            print("DTW/TDI 계산을 위한 유효한 데이터 없음")
            portfolio_metrics['dtw'] = 1.0
            portfolio_metrics['tdi'] = 1.0
    except Exception as e:
        print(f"DTW/TDI 계산 중 오류: {e}")
        portfolio_metrics['dtw'] = 1.0
        portfolio_metrics['tdi'] = 1.0

    result = {'portfolio': portfolio_metrics, 'by_ticker': ticker_results}
    avg_sharpe = np.mean([ticker_results[ticker_id]['sharpe_ratio'] for ticker_id in unique_tickers])
    result['avg_ticker_sharpe'] = avg_sharpe

    return result


def backtest_buy_and_hold(actual_returns, ticker_ids, commission=DEFAULT_COMMISSION, risk_free_rate=None):
    """Buy and Hold 전략 백테스트 함수"""
    if risk_free_rate is None:
        risk_free_rate = get_risk_free_rate()

    if hasattr(actual_returns, 'values'):
        actual_returns = actual_returns.values
    if hasattr(ticker_ids, 'values'):
        ticker_ids = ticker_ids.values

    actual_returns = np.asarray(actual_returns)
    ticker_ids = np.asarray(ticker_ids)

    unique_tickers = np.unique(ticker_ids)
    n_tickers = len(unique_tickers)
    n_samples = len(actual_returns)

    initial_capital = 1.0
    portfolio_values = [initial_capital]
    daily_returns = []

    ticker_results = {ticker_id: {
        'cash': 0,
        'shares': (initial_capital / n_tickers) * (1 - commission),
        'values': [initial_capital / n_tickers],
        'returns': [],
        'trades': [{
            'day': 0, 'ticker': ticker_id, 'action': 'BUY',
            'shares': (initial_capital / n_tickers) * (1 - commission),
            'price': 1.0, 'value': (initial_capital / n_tickers) * (1 - commission)
        }],
        'last_price': 1.0,
    } for ticker_id in unique_tickers}

    for t in range(n_samples):
        daily_portfolio_value = 0.0
        previous_portfolio_value = portfolio_values[-1]

        for ticker_id in unique_tickers:
            ticker_mask = ticker_ids == ticker_id
            if not any(ticker_mask[t:t+1]):
                ticker_result = ticker_results[ticker_id]
                current_value = ticker_result['shares'] * ticker_result['last_price']
                daily_portfolio_value += current_value
                ticker_result['returns'].append(0.0)
                ticker_result['values'].append(current_value)
                continue

            ticker_actual = actual_returns[t:t+1][ticker_mask[t:t+1]][0]
            ticker_result = ticker_results[ticker_id]
            ticker_result['last_price'] *= (1 + ticker_actual)

            current_value = ticker_result['shares'] * ticker_result['last_price']
            previous_value = ticker_result['values'][-1] if ticker_result['values'] else initial_capital / n_tickers
            ticker_daily_return = (current_value / previous_value) - 1 if previous_value > 0 else 0

            ticker_result['returns'].append(ticker_daily_return)
            ticker_result['values'].append(current_value)
            daily_portfolio_value += current_value

        portfolio_daily_return = (daily_portfolio_value / previous_portfolio_value) - 1 if previous_portfolio_value > 0 else 0
        daily_returns.append(portfolio_daily_return)
        portfolio_values.append(daily_portfolio_value)

    portfolio_metrics = calculate_performance_metrics(portfolio_values, daily_returns, risk_free_rate)

    all_trades = []
    for ticker_id in unique_tickers:
        all_trades.extend(ticker_results[ticker_id]['trades'])
    portfolio_metrics['trades'] = all_trades

    for ticker_id in unique_tickers:
        ticker_values = ticker_results[ticker_id]['values']
        ticker_returns = ticker_results[ticker_id]['returns']

        if len(ticker_returns) > 0:
            ticker_metrics = calculate_performance_metrics(ticker_values, ticker_returns, risk_free_rate)
            ticker_results[ticker_id].update(ticker_metrics)
        else:
            ticker_results[ticker_id].update({
                'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0,
                'trade_count': 1,
                'win_rate': 1.0 if portfolio_metrics['total_return'] > 0 else 0.0,
                'risk_free_rate': risk_free_rate
            })

    result = {'portfolio': portfolio_metrics, 'by_ticker': ticker_results}
    avg_sharpe = np.mean([ticker_results[ticker_id]['sharpe_ratio'] for ticker_id in unique_tickers])
    result['avg_ticker_sharpe'] = avg_sharpe

    return result


def find_optimal_threshold(y_pred_val, y_val_flat, ticker_val_flat, risk_free_rate,
                           min_expected_trades, use_combined_score=True, total_opportunities=None):
    """최적 임계값을 찾습니다. 탐색 범위는 예측값 분포 기반으로 동적 계산."""
    abs_preds = np.abs(y_pred_val)
    t_max = float(np.percentile(abs_preds, 95))
    if t_max < 1e-6:
        t_max = 1e-4
    thresholds = np.linspace(0.0, t_max, THRESHOLD_N_CANDIDATES)

    best_weighted_score = -np.inf
    best_threshold = 0
    best_backtest = None
    all_thresholds = {}

    for threshold in thresholds:
        result = backtest_by_ticker(
            predictions=y_pred_val,
            actual_returns=y_val_flat,
            ticker_ids=ticker_val_flat,
            threshold=threshold,
            commission=DEFAULT_COMMISSION,
            risk_free_rate=risk_free_rate,
        )

        all_thresholds[float(threshold)] = {
            'total_return': result['portfolio']['total_return'],
            'sharpe_ratio': result['portfolio']['sharpe_ratio'],
            'max_drawdown': result['portfolio']['max_drawdown'],
            'trades': result['portfolio'].get('trades', []),
        }

        trade_count = len(result['portfolio'].get('trades', []))

        if use_combined_score:
            weighted_score = calculate_combined_score(
                result,
                total_opportunities=total_opportunities,
            )
        else:
            trade_ratio_score = (
                min(1.0, trade_count / min_expected_trades)
                if trade_count >= min_expected_trades * 0.5
                else (trade_count / min_expected_trades) ** 2
            )
            weighted_score = result['avg_ticker_sharpe'] * trade_ratio_score

        if weighted_score > best_weighted_score:
            best_weighted_score = weighted_score
            best_threshold = threshold
            best_backtest = result

    return best_threshold, best_backtest, all_thresholds

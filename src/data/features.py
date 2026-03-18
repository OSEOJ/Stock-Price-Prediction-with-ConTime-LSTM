"""
기술적 지표 계산 및 파라미터 최적화 모듈
"""
import numpy as np
import pandas as pd
import yfinance as yf

from ..evaluate import get_risk_free_rate


# ─────────────────────────────────────────────
# 기술적 지표 계산 (technical_indicators)
# ─────────────────────────────────────────────

def calculate_ema_series(series, span):
    """주어진 Series에 대해 지수이동평균(EMA)을 계산"""
    return series.ewm(span=span, adjust=False).mean()


def calculate_macd(data, short_span, long_span, signal_span):
    """MACD, Signal Line 계산"""
    ema_short = calculate_ema_series(data['Close'], short_span)
    ema_long = calculate_ema_series(data['Close'], long_span)
    macd = ema_short - ema_long
    signal = calculate_ema_series(macd, signal_span)
    return macd, signal


def calculate_cmf(data, period):
    """Chaikin Money Flow 계산"""
    high = data['High']
    low = data['Low']
    close = data['Close']
    volume = data['Volume']

    mf_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, 0.0001)
    mf_volume = mf_multiplier * volume
    cmf = mf_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()

    data[f'CMF_{period}'] = cmf
    return data


def calculate_rsi(data, period=14):
    """RSI 계산"""
    close = data['Close']
    delta = close.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))

    data[f'RSI_{period}'] = rsi
    return data


def add_technical_indicators(df, ema_params, macd_params, cmf_period, rsi_params):
    """주가 데이터에 최적화된 파라미터로 모든 기술적 지표를 추가하는 함수"""
    df['EMA_short'] = calculate_ema_series(df['Close'], ema_params['short'])
    df['EMA_long'] = calculate_ema_series(df['Close'], ema_params['long'])
    df['EMA_diff'] = df['EMA_short'] - df['EMA_long']

    ema_fast = calculate_ema_series(df['Close'], macd_params['fast'])
    ema_slow = calculate_ema_series(df['Close'], macd_params['slow'])
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = calculate_ema_series(df['MACD'], macd_params['signal'])
    df['MACD_diff'] = df['MACD'] - df['MACD_signal']

    df = calculate_cmf(df, cmf_period)
    df['CMF'] = df[f'CMF_{cmf_period}']
    df.drop(columns=[f'CMF_{cmf_period}'], inplace=True)

    df = calculate_rsi(df, rsi_params['period'])
    df['RSI'] = df[f'RSI_{rsi_params["period"]}']
    df.drop(columns=[f'RSI_{rsi_params["period"]}'], inplace=True)

    df['RSI_upper'] = rsi_params['upper_threshold']
    df['RSI_lower'] = rsi_params['lower_threshold']

    df['Return'] = df['Close'].pct_change()

    return df


# ─────────────────────────────────────────────
# 공통 유틸리티
# ─────────────────────────────────────────────

def _get_scalar_rf_rate(data, risk_free_rates=None):
    """무위험 수익률을 스칼라 float으로 변환"""
    if risk_free_rates is None:
        return 0.01
    try:
        if isinstance(risk_free_rates, (float, int)):
            return float(risk_free_rates)
        if hasattr(risk_free_rates, 'index'):
            subset = risk_free_rates.loc[data.index[0]:data.index[-1]]
            return float(subset.mean() if not subset.empty else risk_free_rates.mean())
    except Exception as e:
        print(f"무위험 수익률 처리 오류: {e}")
    return 0.01


def _simulate_portfolio(close_values, positions, n_total):
    """매매 포지션으로부터 포트폴리오 가치를 시뮬레이션합니다."""
    returns = []
    for i, (idx, action) in enumerate(positions[:-1]):
        if action == 'buy':
            sell_idx = positions[i + 1][0]
            returns.append((close_values[sell_idx] - close_values[idx]) / close_values[idx])

    portfolio_values = [1.0]
    position = False
    buy_idx = 0

    for idx, action in positions:
        while len(portfolio_values) <= idx:
            portfolio_values.append(portfolio_values[-1])
        if action == 'buy' and not position:
            position = True
            buy_idx = idx
        elif action == 'sell' and position:
            position = False
            trade_return = (close_values[idx] - close_values[buy_idx]) / close_values[buy_idx]
            portfolio_values[-1] *= (1 + trade_return)

    while len(portfolio_values) < n_total:
        portfolio_values.append(portfolio_values[-1])

    cummax = np.maximum.accumulate(portfolio_values)
    drawdowns = (cummax - portfolio_values) / cummax
    mdd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

    total_return = float(np.prod([1 + r for r in returns]) - 1) if returns else 0
    return total_return, returns, portfolio_values, mdd


def _calc_performance(total_return, returns, mdd, data_length, risk_free_rate):
    """연간수익률·샤프지수·MDD를 계산합니다."""
    annual_return = ((1 + total_return) ** (252 / data_length)) - 1
    annual_std = np.std(returns) * np.sqrt(252) if returns else 0
    sharpe_ratio = (annual_return - risk_free_rate) / annual_std if annual_std != 0 else 0
    return {
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'mdd': mdd,
        'total_return': total_return,
    }


def _minmax(value, min_val, max_val, inverse=False):
    if max_val == min_val:
        return 0
    normalized = (value - min_val) / (max_val - min_val)
    return 1 - normalized if inverse else normalized


def _find_best(all_results, param_keys):
    """샤프지수(90%) + MDD역(10%) 기준으로 최적 파라미터를 반환합니다."""
    if not all_results:
        return None
    sharpes = [r['metrics']['sharpe_ratio'] for r in all_results]
    mdds = [r['metrics']['mdd'] for r in all_results]
    min_s, max_s = min(sharpes), max(sharpes)
    min_m, max_m = min(mdds), max(mdds)

    best_score, best_params = float('-inf'), None
    for result in all_results:
        score = (0.9 * _minmax(result['metrics']['sharpe_ratio'], min_s, max_s) +
                 0.1 * _minmax(result['metrics']['mdd'], min_m, max_m, inverse=True))
        if score > best_score:
            best_score = score
            best_params = {k: result[k] for k in param_keys}
            best_params['metrics'] = result['metrics']
    return best_params


def _average_params(params_list, keys, defaults):
    """파라미터 리스트의 정수 평균을 반환합니다."""
    if not params_list:
        return defaults
    return {k: int(np.mean([p[k] for p in params_list])) for k in keys}


# ─────────────────────────────────────────────
# 지표별 백테스트
# ─────────────────────────────────────────────

def backtest_ema(data, short_ema, long_ema):
    """EMA 크로스오버 백테스팅"""
    close_values = data['Close'].values
    sv, lv = short_ema.values, long_ema.values

    positions = []
    for i in range(1, len(sv)):
        if sv[i-1] <= lv[i-1] and sv[i] > lv[i]:
            positions.append((i, 'buy'))
        elif sv[i-1] >= lv[i-1] and sv[i] < lv[i]:
            positions.append((i, 'sell'))

    return _simulate_portfolio(close_values, positions, len(close_values))


def backtest_macd(data, macd, signal):
    """MACD 크로스오버 백테스팅"""
    close_values = data['Close'].values
    mv, sv = macd.values, signal.values

    positions = []
    for i in range(1, len(mv)):
        if mv[i-1] <= sv[i-1] and mv[i] > sv[i]:
            positions.append((i, 'buy'))
        elif mv[i-1] >= sv[i-1] and mv[i] < sv[i]:
            positions.append((i, 'sell'))

    return _simulate_portfolio(close_values, positions, len(close_values))


def backtest_cmf(data, cmf, threshold=0.05):
    """CMF 임계값 백테스팅"""
    close_values = data['Close'].values
    cv = cmf.values

    positions = []
    for i in range(1, len(cv)):
        if np.isnan(cv[i-1]) or np.isnan(cv[i]):
            continue
        if cv[i-1] <= threshold and cv[i] > threshold:
            positions.append((i, 'buy'))
        elif cv[i-1] >= -threshold and cv[i] < -threshold:
            positions.append((i, 'sell'))

    return _simulate_portfolio(close_values, positions, len(close_values))


def backtest_rsi(data, rsi, upper_threshold, lower_threshold):
    """RSI 과매도/과매수 백테스팅"""
    close_values = data['Close'].values
    rv = rsi.values

    positions = []
    for i in range(1, len(rv)):
        if np.isnan(rv[i-1]) or np.isnan(rv[i]):
            continue
        if rv[i-1] <= lower_threshold and rv[i] > lower_threshold:
            positions.append((i, 'buy'))
        elif rv[i-1] >= upper_threshold and rv[i] < upper_threshold:
            positions.append((i, 'sell'))

    return _simulate_portfolio(close_values, positions, len(close_values))


# ─────────────────────────────────────────────
# 지표별 전략 평가 및 파라미터 최적화
# ─────────────────────────────────────────────

def evaluate_ema_strategy(data, short_period, long_period, risk_free_rates=None):
    short_ema = calculate_ema_series(data['Close'], short_period)
    long_ema = calculate_ema_series(data['Close'], long_period)
    total_return, returns, portfolio_values, mdd = backtest_ema(data, short_ema, long_ema)
    if not returns:
        return {'annual_return': 0, 'sharpe_ratio': 0, 'mdd': 1, 'total_return': 0}
    metrics = _calc_performance(total_return, returns, mdd, len(data),
                                _get_scalar_rf_rate(data, risk_free_rates))
    metrics.update({'short_period': short_period, 'long_period': long_period})
    return metrics


def optimize_ema_parameters(data, risk_free_rates=None):
    all_results = []
    for short in range(5, 50, 5):
        for long in range(50, 200, 10):
            if short >= long:
                continue
            try:
                all_results.append({
                    'short': short, 'long': long,
                    'metrics': evaluate_ema_strategy(data, short, long, risk_free_rates)
                })
            except Exception as e:
                print(f"EMA {short}-{long} 오류: {e}")
    return _find_best(all_results, ['short', 'long']) or {'short': 10, 'long': 50}


def evaluate_macd_strategy(data, fast_period, slow_period, signal_period, risk_free_rates=None):
    macd, signal = calculate_macd(data, fast_period, slow_period, signal_period)
    total_return, returns, portfolio_values, mdd = backtest_macd(data, macd, signal)
    if not returns:
        return {'annual_return': 0, 'sharpe_ratio': 0, 'mdd': 1, 'total_return': 0}
    metrics = _calc_performance(total_return, returns, mdd, len(data),
                                _get_scalar_rf_rate(data, risk_free_rates))
    metrics.update({'fast_period': fast_period, 'slow_period': slow_period, 'signal_period': signal_period})
    return metrics


def optimize_macd_parameters(data, risk_free_rates=None):
    all_results = []
    for fast in range(5, 20, 2):
        for slow in range(20, 60, 5):
            if fast >= slow:
                continue
            for signal in range(5, 20, 2):
                try:
                    all_results.append({
                        'fast': fast, 'slow': slow, 'signal': signal,
                        'metrics': evaluate_macd_strategy(data, fast, slow, signal, risk_free_rates)
                    })
                except Exception as e:
                    print(f"MACD {fast}-{slow}-{signal} 오류: {e}")
    return _find_best(all_results, ['fast', 'slow', 'signal']) or {'fast': 12, 'slow': 26, 'signal': 9}


def evaluate_cmf_strategy(data, period, threshold=0.05, risk_free_rates=None):
    df_temp = calculate_cmf(data.copy(), period)
    cmf = df_temp[f'CMF_{period}']
    total_return, returns, portfolio_values, mdd = backtest_cmf(data, cmf, threshold)
    if not returns:
        return {'annual_return': 0, 'sharpe_ratio': 0, 'mdd': 1, 'total_return': 0}
    metrics = _calc_performance(total_return, returns, mdd, len(data),
                                _get_scalar_rf_rate(data, risk_free_rates))
    metrics.update({'period': period})
    return metrics


def optimize_cmf_period(data, risk_free_rates=None):
    all_results = []
    for period in range(10, 50, 5):
        try:
            all_results.append({
                'period': period,
                'metrics': evaluate_cmf_strategy(data, period, risk_free_rates=risk_free_rates)
            })
        except Exception as e:
            print(f"CMF {period} 오류: {e}")
    best = _find_best(all_results, ['period'])
    return best['period'] if best else 20


def evaluate_rsi_strategy(data, period, upper_threshold, lower_threshold, risk_free_rates=None):
    df_temp = calculate_rsi(data.copy(), period)
    rsi = df_temp[f'RSI_{period}']
    total_return, returns, portfolio_values, mdd = backtest_rsi(data, rsi, upper_threshold, lower_threshold)
    if not returns:
        return {'annual_return': 0, 'sharpe_ratio': 0, 'mdd': 1, 'total_return': 0}
    metrics = _calc_performance(total_return, returns, mdd, len(data),
                                _get_scalar_rf_rate(data, risk_free_rates))
    metrics.update({'period': period, 'upper_threshold': upper_threshold, 'lower_threshold': lower_threshold})
    return metrics


def optimize_rsi_parameters(data, risk_free_rates=None):
    all_results = []
    for period in range(5, 30, 2):
        for upper in range(65, 85, 5):
            for lower in range(15, 35, 5):
                try:
                    all_results.append({
                        'period': period, 'upper_threshold': upper, 'lower_threshold': lower,
                        'metrics': evaluate_rsi_strategy(data, period, upper, lower, risk_free_rates)
                    })
                except Exception as e:
                    print(f"RSI {period}-{upper}-{lower} 오류: {e}")
    return (_find_best(all_results, ['period', 'upper_threshold', 'lower_threshold'])
            or {'period': 14, 'upper_threshold': 70, 'lower_threshold': 30})


# ─────────────────────────────────────────────
# 전체 최적화 실행
# ─────────────────────────────────────────────

def run_technical_optimization(tickers, start_date, end_date):
    """여러 종목에 대해 기술적 지표 최적화를 실행하고 평균 파라미터를 반환합니다."""
    print("주식 데이터 다운로드 및 기술적 지표 최적화 중...")

    try:
        risk_free_rate = get_risk_free_rate(start_date=start_date, end_date=end_date)
        print(f"무위험 수익률: {risk_free_rate:.4f}")
    except Exception as e:
        print(f"무위험 수익률 로드 오류: {e}, 기본값 0.01 사용")
        risk_free_rate = 0.01

    ema_list, macd_list, cmf_list, rsi_list = [], [], [], []

    for ticker in tickers:
        print(f"\n{ticker} 최적화 중...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if len(df) < 100:
                print(f"{ticker}: 데이터 부족")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            for name, fn, lst, default in [
                ('EMA',  lambda d: optimize_ema_parameters(d, risk_free_rate),  ema_list,  {'short': 10, 'long': 50}),
                ('MACD', lambda d: optimize_macd_parameters(d, risk_free_rate), macd_list, {'fast': 12, 'slow': 26, 'signal': 9}),
                ('RSI',  lambda d: optimize_rsi_parameters(d, risk_free_rate),  rsi_list,  {'period': 14, 'upper_threshold': 70, 'lower_threshold': 30}),
            ]:
                try:
                    result = fn(df.copy())
                    lst.append(result)
                    print(f"  {name} 완료: {result}")
                except Exception as e:
                    print(f"  {name} 오류: {e}")
                    lst.append(default)

            try:
                cmf_period = optimize_cmf_period(df.copy(), risk_free_rates=risk_free_rate)
                cmf_list.append(cmf_period)
                print(f"  CMF 완료: 기간={cmf_period}")
            except Exception as e:
                print(f"  CMF 오류: {e}")
                cmf_list.append(20)

        except Exception as e:
            print(f"{ticker} 처리 오류: {e}")

    ema_params  = _average_params(ema_list,  ['short', 'long'],                               {'short': 10, 'long': 50})
    macd_params = _average_params(macd_list, ['fast', 'slow', 'signal'],                     {'fast': 12, 'slow': 26, 'signal': 9})
    rsi_params  = _average_params(rsi_list,  ['period', 'upper_threshold', 'lower_threshold'], {'period': 14, 'upper_threshold': 70, 'lower_threshold': 30})
    avg_cmf     = int(np.mean(cmf_list)) if cmf_list else 20

    optimal_params = {
        'ema':  ema_params,
        'macd': macd_params,
        'cmf':  avg_cmf,
        'rsi':  rsi_params,
    }

    print("\n===== 최적화된 파라미터 =====")
    print(f"EMA : 단기={ema_params['short']}, 장기={ema_params['long']}")
    print(f"MACD: 빠름={macd_params['fast']}, 느림={macd_params['slow']}, 신호={macd_params['signal']}")
    print(f"CMF : 기간={avg_cmf}")
    print(f"RSI : 기간={rsi_params['period']}, 상한={rsi_params['upper_threshold']}, 하한={rsi_params['lower_threshold']}")

    return optimal_params

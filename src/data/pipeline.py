"""
데이터 전처리 파이프라인 모듈
(정규화 유틸리티 / 시간 유틸리티 / 시퀀스 준비 / 전체 파이프라인 오케스트레이션)
"""
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .collect import get_economic_data, get_industry_data, add_industry_encoding, combine_stocks_for_embedding
from .features import add_technical_indicators, run_technical_optimization
import os
from dotenv import load_dotenv
from ..config import DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO, DEFAULT_TEST_RATIO, DEFAULT_SPLINE_POINTS

load_dotenv()


# ─────────────────────────────────────────────
# 정규화 유틸리티 (normalize)
# ─────────────────────────────────────────────

def clean_numeric_data(X, replace_nan=0.0, replace_inf=0.0, verbose=False):
    """입력 데이터를 수치형으로 변환하고 이상값 처리"""
    if X is None or X.size == 0:
        return X

    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
        was_1d = True
        was_3d = False
        original_shape = None
    elif X.ndim == 3:
        original_shape = X.shape
        X = X.reshape(X.shape[0], -1)
        was_3d = True
        was_1d = False
    else:
        was_1d = False
        was_3d = False
        original_shape = None

    if np.issubdtype(X.dtype, np.number):
        X_cleaned = X.astype(np.float32)

        nan_mask = np.isnan(X_cleaned)
        inf_mask = np.isinf(X_cleaned)

        if verbose and (nan_mask.any() or inf_mask.any()):
            nan_count = nan_mask.sum()
            inf_count = inf_mask.sum()
            total = X_cleaned.size
            print(f"NaN: {nan_count}개, Inf: {inf_count}개 / 전체 {total}개 ({(nan_count+inf_count)/total*100:.2f}%)")

        X_cleaned = np.nan_to_num(X_cleaned, nan=replace_nan, posinf=replace_inf, neginf=-replace_inf)

    else:
        X_cleaned = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)

        for col in range(X.shape[1]):
            try:
                X_cleaned[:, col] = X[:, col].astype(np.float32)
            except (ValueError, TypeError):
                if verbose:
                    print(f"경고: 열 {col}에 비수치 데이터가 포함되어 있어 인코딩합니다.")
                col_data = X[:, col]
                str_data = [str(x) for x in col_data.flatten()]
                unique_vals = list(set(str_data))
                val_map = {val: i for i, val in enumerate(unique_vals)}
                for i in range(X.shape[0]):
                    X_cleaned[i, col] = float(val_map.get(str(X[i, col]), 0))

        nan_mask = np.isnan(X_cleaned)
        inf_mask = np.isinf(X_cleaned)

        if verbose and (nan_mask.any() or inf_mask.any()):
            nan_count = nan_mask.sum()
            inf_count = inf_mask.sum()
            total = X_cleaned.size
            print(f"NaN: {nan_count}개, Inf: {inf_count}개 / 전체 {total}개 ({(nan_count+inf_count)/total*100:.2f}%)")

        X_cleaned = np.nan_to_num(X_cleaned, nan=replace_nan, posinf=replace_inf, neginf=-replace_inf)

    if was_1d:
        X_cleaned = X_cleaned.flatten()
    elif was_3d:
        X_cleaned = X_cleaned.reshape(original_shape)

    return X_cleaned


def tanh_scale(X, replace_nan=0.0, replace_inf=0.0, verbose=False):
    """Tanh 스케일링 적용: 데이터 정리 후 [-1, 1] 범위로 변환"""
    X_cleaned = clean_numeric_data(X, replace_nan, replace_inf, verbose)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cleaned)
    return np.tanh(X_scaled), scaler



# ─────────────────────────────────────────────
# 시간 유틸리티 (time_utils)
# ─────────────────────────────────────────────

def calculate_time_derivative(x, dt=None, smooth=False):
    """시퀀스 데이터의 시간 도함수 계산 (중앙 차분법)"""
    if len(x) <= 1:
        return np.zeros_like(x)

    dx = np.zeros_like(x)
    is_dt_array = isinstance(dt, (list, np.ndarray)) and len(dt) > 1

    if x.shape[0] > 2:
        if is_dt_array:
            for i in range(1, len(x)-1):
                dt_prev = dt[i-1]
                dt_next = dt[i] if i < len(dt) else 1.0
                total_dt = dt_prev + dt_next
                if total_dt > 0:
                    w_prev = dt_next / total_dt
                    w_next = dt_prev / total_dt
                    dx[i] = (w_next * (x[i+1] - x[i]) / dt_next -
                             w_prev * (x[i] - x[i-1]) / dt_prev)
        else:
            dt_val = 1.0 if dt is None else dt
            dx[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt_val)

    if is_dt_array:
        dt_first = dt[0] if len(dt) > 0 else 1.0
        dx[0] = (x[1] - x[0]) / dt_first
        dt_last = dt[-1] if len(dt) > 0 else 1.0
        dx[-1] = (x[-1] - x[-2]) / dt_last
    else:
        dt_val = 1.0 if dt is None else dt
        dx[0] = (x[1] - x[0]) / dt_val
        dx[-1] = (x[-1] - x[-2]) / dt_val

    if smooth and len(x) > 3:
        kernel = np.array([0.25, 0.5, 0.25])
        dx[1:-1] = np.convolve(dx, kernel, mode='same')[1:-1]

    return dx


def hermite_cubic_spline(data, n_interpolation_points=10, time_points=None):
    """SciPy의 PCHIP 보간기를 사용한 최적화된 스플라인 보간"""
    if time_points is not None:
        original_times = time_points
        total_time = original_times[-1] - original_times[0]
        interp_times = np.linspace(original_times[0], original_times[-1],
                                   int(total_time * n_interpolation_points))
    else:
        original_times = np.arange(len(data))
        interp_times = np.linspace(0, len(data)-1, (len(data)-1)*n_interpolation_points + 1)

    def interpolate_column(col):
        return PchipInterpolator(original_times, data[:, col])(interp_times)

    interpolated_data = Parallel(n_jobs=-1)(
        delayed(interpolate_column)(col)
        for col in range(data.shape[1])
    )

    return np.column_stack(interpolated_data), interp_times


# ─────────────────────────────────────────────
# 시퀀스 데이터 준비 (processors)
# ─────────────────────────────────────────────

def process_data(data, use_spline=True, n_interpolation_points=DEFAULT_SPLINE_POINTS):
    """주식 데이터 전처리 함수"""
    ticker_encoder = LabelEncoder()
    all_tickers = data['ticker'].unique()
    ticker_encoder.fit(all_tickers)
    data['ticker_id'] = ticker_encoder.transform(data['ticker'])

    data = data.ffill()

    ticker_data = {}
    for ticker in all_tickers:
        ticker_df = data[data['ticker'] == ticker].copy()
        ticker_data[ticker] = ticker_df

        if use_spline and n_interpolation_points > 0:
            time_points = ticker_df.index.astype(np.int64) // 10**9
            numeric_cols = ticker_df.select_dtypes(include=[np.number]).columns
            numeric_data = ticker_df[numeric_cols].values

            interpolated_data, interp_times = hermite_cubic_spline(
                numeric_data,
                n_interpolation_points=n_interpolation_points,
                time_points=time_points
            )

            ticker_data[f"{ticker}_spline"] = {
                'data': interpolated_data,
                'times': interp_times,
                'columns': numeric_cols
            }

            print(f"{ticker}: 원본 데이터 {len(ticker_df)}개 → 보간 후 {len(interpolated_data)}개 포인트")

    return data, ticker_encoder, ticker_data


def load_stock_data(ticker_string):
    """티커 문자열로부터 주식 데이터를 로드합니다."""
    ticker_string = ticker_string[:-5] if ticker_string.endswith('_data') else ticker_string
    training_tickers = ticker_string.split('_')
    all_tickers = '_'.join(training_tickers)

    filename = f"./output/raw/{all_tickers}_data.csv"
    data = pd.read_csv(filename, parse_dates=['Date'])
    data = data.set_index('Date')
    data.sort_index(inplace=True)

    return data, training_tickers


def prepare_data(data, window_size=60, n_interpolation_points=None):
    """시계열 데이터를 윈도우 기반 시퀀스로 준비"""
    ticker_encoder = LabelEncoder()
    all_tickers = data['ticker'].unique()
    ticker_encoder.fit(all_tickers)
    data['ticker_id'] = ticker_encoder.transform(data['ticker'])

    data['log_return'] = data.groupby('ticker')['Close'].transform(lambda x: np.log(x).diff())
    data['log_return'] = data['log_return'].fillna(0)

    x_train_list, y_train_list, ticker_train_list, dt_train_list = [], [], [], []
    x_val_list, y_val_list, ticker_val_list, dt_val_list = [], [], [], []
    x_test_list, y_test_list, ticker_test_list, dt_test_list = [], [], [], []

    scalers = {}

    for ticker in data['ticker'].unique():
        ticker_df = data[data['ticker']==ticker]
        ticker_df = ticker_df.sort_index()
        ticker_df['days_diff'] = (ticker_df.index.to_series().diff().dt.days).fillna(1.0)

        drop_columns = ['ticker', 'Close', 'days_diff', 'ticker_id', 'log_return']
        drop_columns = [col for col in drop_columns if col in ticker_df.columns]

        feature_cols = [col for col in ticker_df.columns if col not in drop_columns]
        if len(feature_cols) > 0:
            scaled_features, scaler = tanh_scale(
                ticker_df[feature_cols].values,
                verbose=False
            )
            scalers[ticker] = {
                'scaler': scaler,
                'feature_cols': feature_cols
            }
            features = scaled_features
        else:
            features = np.array([])

        labels = ticker_df[['log_return']].values
        ids = ticker_df['ticker_id'].values
        time_diffs = ticker_df['days_diff'].values

        seq_X, seq_Y, seq_ID, seq_dt = [], [], [], []
        for i in range(len(features) - window_size):
            seq_X.append(features[i:i+window_size])
            seq_Y.append(labels[i+window_size])
            seq_ID.append(ids[i+window_size])
            seq_dt.append(time_diffs[i+1:i+window_size+1])

        if not seq_X:
            continue

        seq_X = np.stack(seq_X)
        seq_Y = np.stack(seq_Y)
        seq_ID = np.array(seq_ID)
        seq_dt = np.stack(seq_dt)

        n = len(seq_X)
        test_size = int(n * DEFAULT_TEST_RATIO)
        val_size = int(n * DEFAULT_VAL_RATIO)
        train_end = n - test_size - val_size

        x_train_list.append(seq_X[:train_end])
        y_train_list.append(seq_Y[:train_end])
        ticker_train_list.append(seq_ID[:train_end])
        dt_train_list.append(seq_dt[:train_end])

        if val_size > 0:
            x_val_list.append(seq_X[train_end:train_end+val_size])
            y_val_list.append(seq_Y[train_end:train_end+val_size])
            ticker_val_list.append(seq_ID[train_end:train_end+val_size])
            dt_val_list.append(seq_dt[train_end:train_end+val_size])

        if test_size > 0:
            x_test_list.append(seq_X[-test_size:])
            y_test_list.append(seq_Y[-test_size:])
            ticker_test_list.append(seq_ID[-test_size:])
            dt_test_list.append(seq_dt[-test_size:])

    if not x_train_list:
        raise ValueError("데이터 준비 중 오류: 학습 데이터가 없습니다.")

    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    ticker_train = np.concatenate(ticker_train_list, axis=0)
    time_diffs_train = np.concatenate(dt_train_list, axis=0)

    if x_val_list:
        x_val = np.concatenate(x_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        ticker_val = np.concatenate(ticker_val_list, axis=0)
        time_diffs_val = np.concatenate(dt_val_list, axis=0)
    else:
        x_val = np.empty((0, window_size, x_train.shape[2]))
        y_val = np.empty((0, 1))
        ticker_val = np.empty((0,))
        time_diffs_val = np.empty((0, window_size))

    if x_test_list:
        x_test = np.concatenate(x_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        ticker_test = np.concatenate(ticker_test_list, axis=0)
        time_diffs_test = np.concatenate(dt_test_list, axis=0)
    else:
        x_test = np.empty((0, window_size, x_train.shape[2]))
        y_test = np.empty((0, 1))
        ticker_test = np.empty((0,))
        time_diffs_test = np.empty((0, window_size))

    y_train_dt = calculate_time_derivative(y_train)
    y_val_dt = calculate_time_derivative(y_val) if len(y_val) > 0 else None
    y_test_dt = calculate_time_derivative(y_test) if len(y_test) > 0 else None

    date_min = data.index.min()
    date_max = data.index.max()

    print(f"전처리 완료: 특성 수={x_train.shape[2]}, 학습 샘플 수={x_train.shape[0]}")

    result_dict = {
        'x_train': x_train, 'y_train': y_train,
        'ticker_train': ticker_train, 'y_train_dt': y_train_dt,
        'time_diffs_train': time_diffs_train,

        'x_val': x_val, 'y_val': y_val,
        'ticker_val': ticker_val, 'y_val_dt': y_val_dt,
        'time_diffs_val': time_diffs_val,

        'x_test': x_test, 'y_test': y_test,
        'ticker_test': ticker_test, 'y_test_dt': y_test_dt,
        'time_diffs_test': time_diffs_test,

        'start_date': date_min,
        'end_date': date_max,
        'scalers': scalers,
        'data': data
    }

    return result_dict, ticker_encoder, data


# ─────────────────────────────────────────────
# 전체 데이터 파이프라인 (data_integration)
# ─────────────────────────────────────────────

def process_stock_data(tickers, start_date, end_date, fred_api_key=None):
    """주식 데이터 처리의 전체 파이프라인을 실행하는 함수"""
    if fred_api_key is None:
        fred_api_key = os.getenv('FRED_API_KEY')
        if not fred_api_key:
            raise EnvironmentError("FRED_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

    print(f"주식 데이터 처리 시작: {len(tickers)}개 종목")

    # 1. 기술적 지표 최적화
    optimal_params = run_technical_optimization(tickers, start_date, end_date)

    # 2. 최적화된 파라미터로 데이터셋 생성
    all_data = {}

    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df_with_indicators = add_technical_indicators(
            df.copy(),
            ema_params=optimal_params['ema'],
            macd_params=optimal_params['macd'],
            cmf_period=optimal_params['cmf'],
            rsi_params=optimal_params['rsi']
        )

        df_with_indicators = df_with_indicators.dropna()
        all_data[ticker] = df_with_indicators
        print(f"{ticker} 데이터 처리 완료: {len(df_with_indicators)}행")

    # 3. 재무제표 데이터 처리 생략 (ETF/지수 대상으로 재무제표 없음)
    financial_data_all = {}
    selected_common_features = []

    # 4. 경제 지표 데이터 추가
    print("\n===== 경제 지표 데이터 처리 시작 =====\n")
    econ_df = get_economic_data(start_date, end_date, fred_api_key)

    # 5. 데이터 최종 통합
    print("\n===== 최종 데이터 통합 =====\n")


    for ticker in tickers:
        if ticker in all_data:
            try:
                stock_data = all_data[ticker].copy()

                if ticker in financial_data_all and selected_common_features:
                    fin_data = financial_data_all[ticker][selected_common_features]
                    stock_data = stock_data.join(fin_data, how='left')

                stock_data = stock_data.join(econ_df, how='left')

                for col in stock_data.columns:
                    if col != 'Close' and stock_data[col].isna().any():
                        stock_data[col] = stock_data[col].interpolate(method='linear')

                stock_data = stock_data.dropna()
                all_data[ticker] = stock_data

                print(f"{ticker} 데이터 통합 완료: {stock_data.shape[1]}개 특성")

            except Exception as e:
                print(f"{ticker} 데이터 통합 실패: {e}")

    # 6. 산업 정보 및 임베딩
    industry_df = get_industry_data(tickers)
    combined_data = combine_stocks_for_embedding(all_data, tickers)
    final_data, industry_encoders = add_industry_encoding(combined_data, industry_df)

    return final_data, all_data, industry_encoders

"""
외부 데이터 수집 모듈 (재무제표 / 거시경제 / 섹터·산업)
"""
import pandas as pd
import numpy as np
import warnings
import yfinance as yf
import FinanceDataReader as fdr
from fredapi import Fred
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", message="Optimization failed to converge")


# ─────────────────────────────────────────────
# 재무제표 데이터 (financial_data)
# ─────────────────────────────────────────────

def _interpolate_financial(df, end_date):
    """재무 데이터를 일별로 보간하고 필요시 미래 데이터 예측"""
    df.index = pd.to_datetime(df.index)

    daily_df = df.resample('D').asfreq()
    for column in daily_df.columns:
        daily_df[column] = daily_df[column].interpolate(method='linear')

    end_date = pd.to_datetime(end_date)
    forecast_steps = (end_date - daily_df.index[-1]).days

    if forecast_steps > 0:
        print(f"예측 시작: {forecast_steps}일")
        date_range = pd.date_range(daily_df.index[-1] + pd.Timedelta(days=1), end_date)

        forecasts = {}
        for column in daily_df.columns:
            try:
                model = ExponentialSmoothing(
                    daily_df[column], trend='add', seasonal=None, seasonal_periods=4
                ).fit()
                forecasts[column] = model.forecast(steps=forecast_steps)
            except Exception as e:
                print(f"{column} 예측 실패: {e}")
                forecasts[column] = np.full(forecast_steps, np.nan)

        forecast_df = pd.DataFrame(forecasts, index=date_range)
        daily_df = pd.concat([daily_df, forecast_df])

    daily_df = daily_df.dropna(axis=1, how='any')
    return daily_df


def process_financial_data(ticker, all_data, stock_end_date):
    """재무제표 데이터를 처리하는 메인 함수 (yfinance 기반)"""
    try:
        print(f"===== {ticker} 재무데이터 처리 시작 =====")

        yf_ticker = yf.Ticker(ticker)

        # 분기별 재무제표 로드 (항목×날짜 → transpose → 날짜×항목)
        income = yf_ticker.quarterly_income_stmt
        balance = yf_ticker.quarterly_balance_sheet
        cashflow = yf_ticker.quarterly_cashflow

        dfs = []
        for name, df in [('Income', income), ('Balance', balance), ('Cashflow', cashflow)]:
            if df is None or df.empty:
                print(f"{ticker}: {name} 데이터를 가져오지 못했습니다.")
                return None
            dfs.append(df.T)

        income_t, balance_t, cashflow_t = dfs

        # ROE 계산
        try:
            ni_col = next((c for c in income_t.columns if 'Net Income' in str(c)), None)
            eq_col = next((c for c in balance_t.columns if 'Stockholders' in str(c) and 'Equity' in str(c)), None)
            if ni_col and eq_col:
                income_t['ROE'] = income_t[ni_col] / balance_t[eq_col]
        except Exception as e:
            print(f"ROE 계산 오류: {e}")

        FS_Summary = pd.concat([income_t, balance_t, cashflow_t], axis=1)
        FS_Summary = FS_Summary.loc[:, ~FS_Summary.columns.duplicated()]
        FS_Summary.index = pd.to_datetime(FS_Summary.index)
        FS_Summary = FS_Summary.sort_index()
        FS_Summary = FS_Summary.apply(pd.to_numeric, errors='coerce')
        FS_Summary = FS_Summary.dropna(axis=1, how='all')

        if FS_Summary.empty:
            print(f"{ticker}: 유효한 재무 데이터가 없습니다.")
            return None

        daily_FS_Summary = _interpolate_financial(FS_Summary, stock_end_date)

        if daily_FS_Summary.empty:
            print(f"{ticker}: 유효한 일별 재무 데이터가 없습니다")
            return None

        if ticker in all_data and 'Close' in all_data[ticker].columns:
            close_df = pd.DataFrame(all_data[ticker]['Close'])
            close_df.columns = ['Close']

            daily_FS_Summary = daily_FS_Summary.merge(
                close_df, left_index=True, right_index=True, how='inner'
            )

            if daily_FS_Summary.empty:
                print(f"{ticker}: 주가 데이터와 병합 후 데이터가 없습니다")
                return None
        else:
            print(f"{ticker}: Close 데이터를 찾을 수 없습니다")
            return None

        print(f"{ticker} 재무 데이터 처리 완료: {daily_FS_Summary.shape}")
        return daily_FS_Summary

    except Exception as e:
        print(f"{ticker} 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────
# 거시경제 지표 데이터 (economic_data)
# ─────────────────────────────────────────────

def _interpolate_and_forecast(df, col_name, stock_end_date):
    """일별 보간 및 지수평활 예측 (단일 컬럼 대상)"""
    df = df.resample('D').asfreq().interpolate()
    forecast_steps = (pd.to_datetime(stock_end_date) - df.index[-1]).days
    if forecast_steps > 0:
        forecast_df = pd.DataFrame(
            index=pd.date_range(df.index[-1] + pd.Timedelta(days=1), stock_end_date)
        )
        model = ExponentialSmoothing(df[col_name], trend='add').fit()
        forecast_df[col_name] = model.forecast(steps=forecast_steps)
        df = pd.concat([df, forecast_df])
    return df


def get_economic_data(start_date, end_date, fred_api_key):
    """경제 지표와 시장 데이터를 수집하는 함수"""
    fred = Fred(api_key=fred_api_key)

    DGS = pd.concat([
        fred.get_series('DGS2', start_date, end_date),
        fred.get_series('DGS5', start_date, end_date),
        fred.get_series('DGS10', start_date, end_date)
    ], axis=1)
    DGS.columns = ['2-year', '5-year', '10-year']

    T10Y2Y = fdr.DataReader('FRED:T10Y2Y', start_date, end_date)
    VIX = fdr.DataReader('FRED:VIXCLS', start_date, end_date)
    Unemployment_Rate = fdr.DataReader('FRED:UNRATE', start_date, end_date)
    CPI = fdr.DataReader('FRED:CPIAUCSL', start_date, end_date)
    FEDFUNDS = fdr.DataReader('FRED:FEDFUNDS', start_date, end_date)
    GDP = pd.DataFrame(fred.get_series('GDP', start_date, end_date), columns=['GDP'])

    index_tickers = {
        "^DJI": "DJI Close",
        "NDAQ": "NDAQ Close",
        "^GSPC": "SPX Close",
        "^RUT": "RUT Close"
    }

    index_data = {}
    for ticker, name in index_tickers.items():
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df[['Close']].rename(columns={'Close': name})
        index_data[name] = df

    Index_data = pd.concat(index_data.values(), axis=1)

    sectors = {
        "VDE": "Energy", "MXI": "Materials", "VIS": "Industrials",
        "VCR": "Consumer Cyclical", "XLP": "Consumer Staples",
        "VHT": "Health Care", "XLF": "Financials",
        "VGT": "Information Technology", "VOX": "Communication Services",
        "XLU": "Utilities", "VNQ": "Real Estate"
    }

    sector_data = {}
    for etf, sector_name in sectors.items():
        df = yf.download(etf, start=start_date, end=end_date)
        df.rename(columns={'Close': f'{sector_name} Close'}, inplace=True)
        sector_data[sector_name] = df[[f'{sector_name} Close']]

    ETF_data = pd.concat(sector_data.values(), axis=1)

    econ_df = (
        _interpolate_and_forecast(DGS, '2-year', end_date)
        .join(_interpolate_and_forecast(T10Y2Y, 'T10Y2Y', end_date), how='left')
        .join(_interpolate_and_forecast(VIX, 'VIXCLS', end_date), how='left')
        .join(_interpolate_and_forecast(Unemployment_Rate, 'UNRATE', end_date), how='left')
        .join(_interpolate_and_forecast(CPI, 'CPIAUCSL', end_date), how='left')
        .join(_interpolate_and_forecast(FEDFUNDS, 'FEDFUNDS', end_date), how='left')
        .join(_interpolate_and_forecast(GDP, 'GDP', end_date), how='left')
    )

    if isinstance(Index_data.columns, pd.MultiIndex):
        Index_data.columns = Index_data.columns.get_level_values(1)
    if isinstance(ETF_data.columns, pd.MultiIndex):
        ETF_data.columns = ETF_data.columns.get_level_values(1)

    econ_df = econ_df.join(Index_data, how='left').join(ETF_data, how='left')
    econ_df = econ_df.ffill().bfill()

    return econ_df


# ─────────────────────────────────────────────
# 섹터·산업 정보 (hierarchical_embedding)
# ─────────────────────────────────────────────

def combine_stocks_for_embedding(all_data, tickers):
    """종목 임베딩을 위한 데이터 통합"""
    print("\n종목 임베딩을 위한 데이터 통합 중...")

    all_stocks_data = []
    for ticker in tickers:
        if ticker in all_data:
            stock_df = all_data[ticker].copy()
            stock_df['ticker'] = ticker
            all_stocks_data.append(stock_df)
            print(f"{ticker} 데이터: {len(stock_df)}행 처리")

    combined_df = pd.concat(all_stocks_data, axis=0)
    combined_df = combined_df.sort_index().reset_index()

    cols = combined_df.columns.tolist()
    cols.remove('ticker')
    combined_df = combined_df[['ticker'] + cols]

    return combined_df


def get_top_n_by_marketcap(start_date: str, n: int = 10,
                           max_workers: int = 20) -> list:
    """
    현재 시총 기준 S&P 500 상위 n개 종목 반환 (fast_info.market_cap 병렬 수집).
    start_date는 서명 호환성을 위해 유지하며, 시총은 현재 값을 근사치로 사용.
    """
    print("  S&P 500 종목 목록 수집 중...")
    sp500_df = fdr.StockListing('S&P500')
    tickers = sp500_df['Symbol'].dropna().tolist()
    tickers = [t.replace('.', '-') for t in tickers]  # BRK.B → BRK-B

    print(f"  시총 수집 중 ({len(tickers)}개, 병렬 workers={max_workers})...")

    def _fetch(ticker):
        try:
            mc = yf.Ticker(ticker).fast_info.market_cap
            if mc and mc > 0:
                return ticker, float(mc)
        except Exception:
            pass
        return ticker, None

    market_caps = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, mc = future.result()
            if mc:
                market_caps[ticker] = mc

    top_n = sorted(market_caps, key=market_caps.get, reverse=True)[:n]
    print(f"  시총 상위 {n}개: {', '.join(top_n)}")
    return top_n


def get_industry_data(tickers):
    """종목 리스트의 산업 및 섹터 정보를 가져옵니다."""
    industry_data = {}

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            industry_data[ticker] = {
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
            print(f"✓ {ticker}: {industry_data[ticker]['sector']} - {industry_data[ticker]['industry']}")
        except Exception as e:
            print(f"✗ {ticker} 데이터 가져오기 실패: {str(e)}")
            industry_data[ticker] = {'sector': 'Unknown', 'industry': 'Unknown'}

    df = pd.DataFrame.from_dict(industry_data, orient='index')
    df.index.name = 'ticker'
    df.reset_index(inplace=True)

    return df


def add_industry_encoding(data, industry_data):
    """데이터에 산업 정보를 추가하고 인코딩합니다."""
    print("\n산업 정보 인코딩 중...")

    data = data.merge(industry_data, on='ticker', how='left')
    data['sector'] = data['sector'].fillna('Unknown')
    data['industry'] = data['industry'].fillna('Unknown')

    sector_encoder = LabelEncoder()
    industry_encoder = LabelEncoder()

    data['sector_id'] = sector_encoder.fit_transform(data['sector'])
    data['industry_id'] = industry_encoder.fit_transform(data['industry'])

    encoders = {
        'sector_encoder': sector_encoder,
        'industry_encoder': industry_encoder,
        'n_sectors': len(sector_encoder.classes_),
        'n_industries': len(industry_encoder.classes_)
    }

    print(f"섹터 수: {encoders['n_sectors']}, 산업 수: {encoders['n_industries']}")

    return data, encoders

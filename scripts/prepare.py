"""
주식 데이터 수집 및 전처리 스크립트
"""
import sys
import os
import json
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.pipeline import process_stock_data, prepare_data
from src.config import DEFAULT_TICKERS, DEFAULT_LOOKBACK_YEARS, DEFAULT_WINDOW_SIZE


def main():
    today = datetime.today()
    start_date = (today - relativedelta(years=DEFAULT_LOOKBACK_YEARS)).strftime('%Y-%m-%d')
    training_tickers = DEFAULT_TICKERS.split(',')
    end_date = datetime.today().strftime('%Y-%m-%d')

    print(f"데이터 수집 시작: {', '.join(training_tickers)}")
    print(f"기간: {start_date} ~ {end_date}")

    final_data, all_data, industry_encoders = process_stock_data(
        training_tickers, start_date, end_date
    )

    # 데이터셋 요약
    print("\n===== 통합 데이터셋 요약 =====")
    print(f"크기: {final_data.shape[0]}행, {final_data.shape[1]}열")
    if 'ticker' in final_data.columns:
        print(f"종목 수: {final_data['ticker'].nunique()}개")
    if 'sector' in final_data.columns:
        print(f"섹터 수: {final_data['sector'].nunique()}개")
    if 'industry' in final_data.columns:
        print(f"산업 수: {final_data['industry'].nunique()}개")
    print(f"기간: {final_data['Date'].min()} ~ {final_data['Date'].max()}")

    for ticker in training_tickers:
        count = len(final_data[final_data['ticker'] == ticker]) if 'ticker' in final_data.columns else '정보 없음'
        print(f"  - {ticker}: {count}행")

    # 원본 데이터 저장
    all_tickers = '_'.join(training_tickers)
    raw_dir = Path("./output/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    final_data.to_csv(raw_dir / f"{all_tickers}_data.csv", index=False)
    print(f"\n원본 데이터 저장: {raw_dir / f'{all_tickers}_data.csv'}")

    # 전처리
    if 'Date' in final_data.columns:
        final_data = final_data.set_index('Date')
        final_data.sort_index(inplace=True)

    print("\n전처리 및 학습용 데이터 생성 시작...")
    processed_data, ticker_encoder, _ = prepare_data(final_data, window_size=DEFAULT_WINDOW_SIZE)

    sector_map = {}
    industry_map = {}
    if 'sector' in final_data.columns and 'sector_id' in final_data.columns:
        for sector, sector_id in zip(final_data['sector'].unique(), final_data['sector_id'].unique()):
            sector_map[sector] = int(sector_id)
    if 'industry' in final_data.columns and 'industry_id' in final_data.columns:
        for industry, industry_id in zip(final_data['industry'].unique(), final_data['industry_id'].unique()):
            industry_map[industry] = int(industry_id)

    processed_data['sector_map'] = sector_map
    processed_data['industry_map'] = industry_map
    processed_data['feature_count'] = processed_data['x_train'].shape[2]

    # 전처리 데이터 저장
    processed_dir = Path("./output/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    with open(processed_dir / f"{all_tickers}_processed.pkl", 'wb') as f:
        pickle.dump(processed_data, f)

    encoder_info = {
        'ticker_encoder': {str(i): ticker for i, ticker in enumerate(ticker_encoder.classes_)},
        'sector_map': sector_map,
        'industry_map': industry_map,
    }
    with open(processed_dir / f"{all_tickers}_encoder_info.json", 'w') as f:
        json.dump(encoder_info, f)

    metadata = {
        'feature_count': processed_data['feature_count'],
        'window_size': DEFAULT_WINDOW_SIZE,
        'tickers': training_tickers,
        'start_date': str(processed_data.get('start_date', start_date)),
        'end_date': str(processed_data.get('end_date', end_date)),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(processed_dir / f"{all_tickers}_metadata.json", 'w') as f:
        json.dump(metadata, f)

    print("\n===== 전처리 완료 =====")
    print(f"학습: {processed_data['x_train'].shape} | 검증: {processed_data['x_val'].shape} | 테스트: {processed_data['x_test'].shape}")
    print(f"특성 수: {processed_data['feature_count']}")


if __name__ == "__main__":
    main()

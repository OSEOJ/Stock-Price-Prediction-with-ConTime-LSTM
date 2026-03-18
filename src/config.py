"""
설정 관리 모듈
"""

# ─────────────────────────────────────────────
# 데이터 수집 기본값
# ─────────────────────────────────────────────
DEFAULT_TICKERS = "SPY"
DEFAULT_LOOKBACK_YEARS = 10      # START_DATE = 오늘 기준 N년 전

# ─────────────────────────────────────────────
# 데이터 전처리
# ─────────────────────────────────────────────
DEFAULT_WINDOW_SIZE = 60
DEFAULT_SPLINE_POINTS = 5       # Hermite cubic spline 보간 밀도

# 시계열 분할 비율 (합계 = 1.0)
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO   = 0.15
DEFAULT_TEST_RATIO  = 0.15

# ─────────────────────────────────────────────
# 모델 아키텍처 (그리드 서치 대상이 아닌 고정값)
# ─────────────────────────────────────────────
DEFAULT_DROPOUT     = 0.5
DEFAULT_MAX_DT      = 3.0       # per-sample dt 클리핑 상한
DEFAULT_MERGE_MODE  = 'ave'     # BiLSTM 방향 결합 방식 ('ave', 'concat', 'sum', 'mul')

# ─────────────────────────────────────────────
# 학습 (optimizer / scheduler / early stopping)
# ─────────────────────────────────────────────
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE    = 64
DEFAULT_EPOCHS        = 50
DEFAULT_PATIENCE      = 10      # EarlyStopping 기준 epoch 수
WARMUP_EPOCHS         = 5       # 학습률 선형 워밍업 epoch 수
LR_DECAY_STEPS        = 3       # 워밍업 이후 학습률 감쇠 횟수
DEFAULT_LR_FACTOR     = 0.5     # 1회 감쇠 시 적용 비율 (lr *= factor)

# ─────────────────────────────────────────────
# 모델 하이퍼파라미터
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    'hidden_dim':   64,         # hidden state 차원 — 모델 용량
    'dt':           1.0,        # RK4 기준 시간 스텝 스케일
    'ode_steps':    3,          # 타임스텝당 RK4 반복 횟수 — 적분 정밀도
    'value_weight': 0.8,        # 예측값 손실 비중 (도함수 손실 = 1 - value_weight)
}

# ─────────────────────────────────────────────
# 백테스트 / 거래 설정
# ─────────────────────────────────────────────
COMMISSION          = 0.001       # 거래 수수료 비율
MIN_TRADES_RATIO    = 0.1       # 전체 기회 대비 최소 거래 비율
MAX_TRADES_RATIO    = 0.5       # 전체 기회 대비 최대 거래 비율

# 최적 임계값 탐색 후보 수 (범위는 예측값 분포 기반으로 동적 계산)
THRESHOLD_N_CANDIDATES = 100
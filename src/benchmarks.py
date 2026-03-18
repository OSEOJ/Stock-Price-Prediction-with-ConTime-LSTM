"""
벤치마크 모델 (Vanilla BiLSTM, ConTime-GRU) + DeLong AUC 검정

ConTime-GRU 참고: "Addressing Prediction Delays in Time Series Forecasting:
A Continuous GRU Approach with Derivative Regularization" (KDD 2024)
GRU-ODE: dh/dt = (1-z) ⊙ (g - h)
"""
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from .config import (
    DEFAULT_DROPOUT, DEFAULT_MAX_DT, DEFAULT_MERGE_MODE,
    DEFAULT_CONFIG,
)
from .contime import DerivativeLayer

_DEFAULT_HIDDEN_DIM = DEFAULT_CONFIG['hidden_dim']
_DEFAULT_DT         = DEFAULT_CONFIG['dt']
_DEFAULT_ODE_STEPS  = DEFAULT_CONFIG['ode_steps']


# ─────────────────────────────────────────────
# 1. Vanilla BiLSTM
# ─────────────────────────────────────────────

class VanillaLSTM(nn.Module):
    """표준 양방향 LSTM — ODE·임베딩 없음"""

    def __init__(self, seq_len, num_features,
                 hidden_dim=_DEFAULT_HIDDEN_DIM, dropout_rate=DEFAULT_DROPOUT):
        super().__init__()
        self.seq_len   = seq_len
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.lstm       = nn.LSTM(num_features, hidden_dim,
                                  batch_first=True, bidirectional=True)
        self.dense        = nn.Linear(hidden_dim * 2, 64)
        self.elu          = nn.ELU()
        self.dropout      = nn.Dropout(dropout_rate)
        self.value_output = nn.Linear(64, 1)

    def forward(self, x, time_diffs=None):
        b, s, f = x.shape
        x   = self.batch_norm(x.reshape(-1, f)).reshape(b, s, f)
        out, _ = self.lstm(x)
        y   = self.elu(self.dense(out))
        y   = self.dropout(y)
        val = self.value_output(y)                  # (b, s, 1)
        return val, torch.zeros_like(val)


def build_vanilla_lstm(seq_len, num_features,
                       hidden_dim=_DEFAULT_HIDDEN_DIM,
                       dropout_rate=DEFAULT_DROPOUT):
    return VanillaLSTM(seq_len, num_features, hidden_dim, dropout_rate)


# ─────────────────────────────────────────────
# 2. ConTime-GRU  (KDD 2024, CONTIME 논문 기반)
#    GRU-ODE:  dh/dt = (1-z) ⊙ (g - h)
#    z = σ(Wz [x, h]),  r = σ(Wr [x, h])
#    g = tanh(Wg [x, r ⊙ h])
# ─────────────────────────────────────────────

class GRUODEFunc(nn.Module):
    """dh/dt = (1-z) ⊙ (g - h)"""

    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.W_z = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.W_g = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        for layer in [self.W_z, self.W_r, self.W_g]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x_t, h):
        xh = torch.cat([x_t, h], dim=-1)
        z  = torch.sigmoid(self.W_z(xh))
        r  = torch.sigmoid(self.W_r(xh))
        g  = torch.tanh(self.W_g(torch.cat([x_t, r * h], dim=-1)))
        return (1.0 - z) * (g - h)


class ContinuousGRULayer(nn.Module):
    """단방향 ODE-GRU (RK4, cell state 없음)"""

    def __init__(self, feature_dim, hidden_dim=_DEFAULT_HIDDEN_DIM,
                 dt=_DEFAULT_DT, ode_steps=_DEFAULT_ODE_STEPS,
                 max_dt=DEFAULT_MAX_DT, reverse=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ode_steps  = ode_steps
        self.max_dt     = max_dt
        self.reverse    = reverse
        self.ode_func   = GRUODEFunc(feature_dim, hidden_dim)

    def _rk4(self, x_t, h, dt):
        k1 = self.ode_func(x_t, h)
        k2 = self.ode_func(x_t, h + 0.5 * dt * k1)
        k3 = self.ode_func(x_t, h + 0.5 * dt * k2)
        k4 = self.ode_func(x_t, h + dt * k3)
        return h + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def forward(self, x, time_diffs):
        if self.reverse:
            x, time_diffs = x.flip(1), time_diffs.flip(1)
        b, seq_len, _ = x.shape
        h = torch.zeros(b, self.hidden_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            sub_dt = (torch.clamp(time_diffs[:, t], max=self.max_dt)
                      / self.ode_steps).unsqueeze(-1)
            for _ in range(self.ode_steps):
                h = self._rk4(x[:, t, :], h, sub_dt)
            outputs.append(h.unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        return out.flip(1) if self.reverse else out


class BidirectionalGRULayer(nn.Module):
    def __init__(self, feature_dim, hidden_dim=_DEFAULT_HIDDEN_DIM,
                 dt=_DEFAULT_DT, ode_steps=_DEFAULT_ODE_STEPS,
                 merge_mode=DEFAULT_MERGE_MODE):
        super().__init__()
        self.merge_mode = merge_mode
        self.fwd = ContinuousGRULayer(feature_dim, hidden_dim, dt, ode_steps, reverse=False)
        self.bwd = ContinuousGRULayer(feature_dim, hidden_dim, dt, ode_steps, reverse=True)

    def forward(self, x, time_diffs):
        f, b = self.fwd(x, time_diffs), self.bwd(x, time_diffs)
        if self.merge_mode == 'concat': return torch.cat([f, b], dim=-1)
        if self.merge_mode == 'sum':    return f + b
        if self.merge_mode == 'mul':    return f * b
        return (f + b) / 2.0    # 'ave' (default)


class ConTimeGRU(nn.Module):
    """
    ConTime-GRU: LSTM ODE → GRU-ODE 교체
    (KDD 2024 CONTIME 논문 구조)
    """

    def __init__(self, seq_len, num_features,
                 hidden_dim=_DEFAULT_HIDDEN_DIM, dropout_rate=DEFAULT_DROPOUT,
                 dt=_DEFAULT_DT, ode_steps=_DEFAULT_ODE_STEPS,
                 merge_mode=DEFAULT_MERGE_MODE):
        super().__init__()
        self.seq_len = seq_len

        self.batch_norm = nn.BatchNorm1d(num_features)

        # ODE-GRU (LSTM 대신)
        self.bigru = BidirectionalGRULayer(
            feature_dim=num_features, hidden_dim=hidden_dim,
            dt=dt, ode_steps=ode_steps, merge_mode=merge_mode,
        )

        self.dense        = nn.Linear(hidden_dim, 64)
        self.elu          = nn.ELU()
        self.dropout      = nn.Dropout(dropout_rate)
        self.value_output = nn.Linear(64, 1)
        self.derivative_layer = DerivativeLayer()

    def forward(self, x, time_diffs):
        b, s, f = x.shape
        x_normed = self.batch_norm(x.reshape(-1, f)).reshape(b, s, f)
        gru_out = self.bigru(x_normed, time_diffs)
        y = self.elu(self.dense(gru_out))
        y = self.dropout(y)
        val = self.value_output(y)
        deriv = self.derivative_layer(val, time_diffs)
        return val, deriv


def build_contime_gru(seq_len, num_features,
                      hidden_dim=_DEFAULT_HIDDEN_DIM,
                      dropout_rate=DEFAULT_DROPOUT,
                      dt=_DEFAULT_DT,
                      ode_steps=_DEFAULT_ODE_STEPS,
                      merge_mode=DEFAULT_MERGE_MODE):
    return ConTimeGRU(seq_len, num_features, hidden_dim, dropout_rate,
                      dt, ode_steps, merge_mode)


# ─────────────────────────────────────────────
# 3. DeLong AUC 검정
# ─────────────────────────────────────────────

def _structural_components(y_true, y_score):
    """DeLong V10, V01 구조적 성분 (벡터화)"""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return 0.5, np.full(max(n1, 1), 0.5), np.full(max(n0, 1), 0.5)
    cmp = pos[:, np.newaxis] - neg[np.newaxis, :]       # (n1, n0)
    V10 = np.mean(cmp > 0, axis=1) + 0.5 * np.mean(cmp == 0, axis=1)
    V01 = np.mean(cmp > 0, axis=0) + 0.5 * np.mean(cmp == 0, axis=0)
    return float(np.mean(V10)), V10, V01


def delong_roc_test(y_true, pred_a, pred_b):
    """
    DeLong test: H0: AUC(pred_a) == AUC(pred_b)

    Parameters
    ----------
    y_true  : array-like, binary (1=상승, 0=하락)
    pred_a  : array-like, 모델 A 연속 예측값
    pred_b  : array-like, 모델 B 연속 예측값

    Returns
    -------
    z_stat, p_value, auc_a, auc_b
    """
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)

    auc_a, V10_a, V01_a = _structural_components(y_true, pred_a)
    auc_b, V10_b, V01_b = _structural_components(y_true, pred_b)

    n1 = int(np.sum(y_true == 1))
    n0 = int(np.sum(y_true == 0))
    if n1 < 2 or n0 < 2:
        return float('nan'), float('nan'), auc_a, auc_b

    # Var(AUC_a - AUC_b) = (S10_aa + S10_bb - 2*S10_ab)/n1
    #                     + (S01_aa + S01_bb - 2*S01_ab)/n0
    mat10 = np.cov(np.vstack([V10_a, V10_b]))   # (2, 2), ddof=1
    mat01 = np.cov(np.vstack([V01_a, V01_b]))   # (2, 2), ddof=1

    var_diff = ((mat10[0, 0] + mat10[1, 1] - 2 * mat10[0, 1]) / n1 +
                (mat01[0, 0] + mat01[1, 1] - 2 * mat01[0, 1]) / n0)
    if var_diff <= 0:
        return 0.0, 1.0, auc_a, auc_b

    z     = (auc_a - auc_b) / np.sqrt(var_diff)
    p_val = float(2 * (1 - stats.norm.cdf(abs(z))))
    return float(z), p_val, auc_a, auc_b

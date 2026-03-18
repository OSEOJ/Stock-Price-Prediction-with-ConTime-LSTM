"""
ConTime 모델 구축
"""
import torch
import torch.nn as nn
from .config import (
    DEFAULT_DROPOUT, DEFAULT_MAX_DT, DEFAULT_MERGE_MODE,
    DEFAULT_CONFIG,
)

_DEFAULT_HIDDEN_DIM = DEFAULT_CONFIG['hidden_dim']
_DEFAULT_DT         = DEFAULT_CONFIG['dt']
_DEFAULT_ODE_STEPS  = DEFAULT_CONFIG['ode_steps']


class ODEFunc(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(ODEFunc, self).__init__()
        self.hidden_dim = hidden_dim
        input_dim = feature_dim + hidden_dim

        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.W_g = nn.Linear(input_dim, hidden_dim)

        nn.init.xavier_uniform_(self.W_i.weight)
        nn.init.xavier_uniform_(self.W_f.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.xavier_uniform_(self.W_g.weight)

    def forward(self, x_t, h, c):
        combined = torch.cat([x_t, h], dim=-1)
        i = torch.sigmoid(self.W_i(combined))
        f = torch.sigmoid(self.W_f(combined))
        o = torch.sigmoid(self.W_o(combined))
        g = torch.tanh(self.W_g(combined))
        dc_dt = i * g + f * c - c
        dh_dt = o * torch.tanh(c) - h
        return dh_dt, dc_dt


class ContinuousLSTMLayer(nn.Module):
    def __init__(self, feature_dim, hidden_dim, return_sequences=True,
                 dt=_DEFAULT_DT, ode_steps=_DEFAULT_ODE_STEPS, max_dt=DEFAULT_MAX_DT, reverse=False):
        super(ContinuousLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        self.dt = dt
        self.ode_steps = ode_steps
        self.max_dt = max_dt
        self.reverse = reverse
        self.ode_func = ODEFunc(feature_dim, hidden_dim)

    def rk4_step(self, x_t, h, c, dt):
        k1_h, k1_c = self.ode_func(x_t, h, c)
        k2_h, k2_c = self.ode_func(x_t, h + 0.5 * dt * k1_h, c + 0.5 * dt * k1_c)
        k3_h, k3_c = self.ode_func(x_t, h + 0.5 * dt * k2_h, c + 0.5 * dt * k2_c)
        k4_h, k4_c = self.ode_func(x_t, h + dt * k3_h, c + dt * k3_c)
        h_new = h + (dt / 6.0) * (k1_h + 2 * k2_h + 2 * k3_h + k4_h)
        c_new = c + (dt / 6.0) * (k1_c + 2 * k2_c + 2 * k3_c + k4_c)
        return h_new, c_new

    def solve_ode(self, x_t, h, c, dt_value):
        # dt_value: (batch,) — 샘플별 시간 간격 (배치 평균 대신 개별 처리)
        adaptive_dt = torch.clamp(dt_value, max=self.max_dt)  # (batch,)
        sub_dt = (adaptive_dt / self.ode_steps).unsqueeze(-1)  # (batch, 1)
        for _ in range(self.ode_steps):
            h, c = self.rk4_step(x_t, h, c, sub_dt)
        return h, c

    def forward(self, x, time_diffs):
        if self.reverse:
            x = x.flip(1)
            time_diffs = time_diffs.flip(1)

        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        c = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            h, c = self.solve_ode(x[:, t, :], h, c, time_diffs[:, t])
            outputs.append(h.unsqueeze(1))

        all_outputs = torch.cat(outputs, dim=1)
        if self.reverse:
            all_outputs = all_outputs.flip(1)

        return all_outputs if self.return_sequences else all_outputs[:, -1, :]


class BidirectionalContinuousLSTMLayer(nn.Module):
    def __init__(self, feature_dim, hidden_dim=_DEFAULT_HIDDEN_DIM, return_sequences=True,
                 dt=_DEFAULT_DT, ode_steps=_DEFAULT_ODE_STEPS, merge_mode=DEFAULT_MERGE_MODE):
        super(BidirectionalContinuousLSTMLayer, self).__init__()
        self.return_sequences = return_sequences
        self.merge_mode = merge_mode

        self.forward_layer = ContinuousLSTMLayer(
            feature_dim, hidden_dim, return_sequences=True, dt=dt, ode_steps=ode_steps, reverse=False
        )
        self.backward_layer = ContinuousLSTMLayer(
            feature_dim, hidden_dim, return_sequences=True, dt=dt, ode_steps=ode_steps, reverse=True
        )

    def forward(self, x, time_diffs):
        fwd = self.forward_layer(x, time_diffs)
        bwd = self.backward_layer(x, time_diffs)

        if self.merge_mode == 'concat':
            merged = torch.cat([fwd, bwd], dim=-1)
        elif self.merge_mode == 'ave':
            merged = (fwd + bwd) / 2.0
        elif self.merge_mode == 'sum':
            merged = fwd + bwd
        elif self.merge_mode == 'mul':
            merged = fwd * bwd
        else:
            merged = torch.cat([fwd, bwd], dim=-1)

        return merged if self.return_sequences else merged[:, -1, :]


class DerivativeLayer(nn.Module):
    def forward(self, value_seq, time_diffs):
        """
        value_seq: (batch, seq, 1)
        time_diffs: (batch, seq)
        반환: (batch, seq, 1)
        """
        values = value_seq[..., 0]  # (batch, seq)
        seq_len = values.shape[1]

        # 1. 첫 번째 지점 (전진 차분)
        first_deriv = (values[:, 1] - values[:, 0]) / torch.clamp(time_diffs[:, 0], min=1e-6)
        first_deriv = first_deriv.unsqueeze(1)  # (batch, 1)

        # 2. 중간 지점 (중앙 차분)
        if seq_len > 2:
            central_diffs = values[:, 2:] - values[:, :-2]
            central_time = time_diffs[:, 1:-1] + time_diffs[:, :-2]
            middle_derivs = central_diffs / torch.clamp(central_time, min=1e-6)  # (batch, seq-2)
        else:
            middle_derivs = torch.zeros(values.shape[0], 0, device=values.device, dtype=values.dtype)

        # 3. 마지막 지점 (후진 차분)
        if seq_len > 1:
            last_deriv = (values[:, -1] - values[:, -2]) / torch.clamp(time_diffs[:, -1], min=1e-6)
            last_deriv = last_deriv.unsqueeze(1)  # (batch, 1)
        else:
            last_deriv = torch.zeros(values.shape[0], 0, device=values.device, dtype=values.dtype)

        derivatives = torch.cat([first_deriv, middle_derivs, last_deriv], dim=1)  # (batch, seq)
        return derivatives.unsqueeze(-1)  # (batch, seq, 1)


class ContimeLSTM(nn.Module):
    def __init__(self, seq_len, num_features, hidden_dim=_DEFAULT_HIDDEN_DIM, dropout_rate=DEFAULT_DROPOUT,
                 dt=_DEFAULT_DT, ode_steps=_DEFAULT_ODE_STEPS,
                 merge_mode=DEFAULT_MERGE_MODE):
        super(ContimeLSTM, self).__init__()
        self.seq_len = seq_len

        # 특성 정규화
        self.batch_norm = nn.BatchNorm1d(num_features)

        # 양방향 연속 LSTM
        self.bilstm = BidirectionalContinuousLSTMLayer(
            feature_dim=num_features,
            hidden_dim=hidden_dim,
            return_sequences=True,
            dt=dt,
            ode_steps=ode_steps,
            merge_mode=merge_mode
        )

        # 출력 레이어
        self.dense = nn.Linear(hidden_dim, 64)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.value_output = nn.Linear(64, 1)

        self.derivative_layer = DerivativeLayer()

    def forward(self, x, time_diffs):
        b, s, f = x.shape
        x_normed = self.batch_norm(x.reshape(-1, f)).reshape(b, s, f)
        lstm_out = self.bilstm(x_normed, time_diffs)
        y_seq = self.elu(self.dense(lstm_out))
        y_seq = self.dropout(y_seq)
        value_output = self.value_output(y_seq)
        derivative_output = self.derivative_layer(value_output, time_diffs)
        return value_output, derivative_output


def build_contime_lstm_model(seq_len, num_features, hidden_dim=_DEFAULT_HIDDEN_DIM,
                              dropout_rate=DEFAULT_DROPOUT,
                              dt=_DEFAULT_DT, ode_steps=_DEFAULT_ODE_STEPS,
                              value_weight=0.8, derivative_weight=0.2,
                              merge_mode=DEFAULT_MERGE_MODE):
    """
    ConTime 모델 생성
    value_weight, derivative_weight는 학습 시 손실 가중치로 사용 (모델 외부에서 관리)
    merge_mode: BiLSTM 방향 결합 방식 ('ave', 'concat', 'sum', 'mul')
    """
    model = ContimeLSTM(
        seq_len=seq_len,
        num_features=num_features,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        dt=dt,
        ode_steps=ode_steps,
        merge_mode=merge_mode,
    )

    return model

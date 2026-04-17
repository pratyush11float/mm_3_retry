"""Power-function EMA + post-hoc EMA reconstruction (EDM2 paper).

Ported from `edm2/training/phema.py` to support consistency-distillation runs.
"""

import copy
import numpy as np
import torch

# -----------------------------------------------------------------------------
# Convert power-function exponent <-> relative standard deviation (s_rel).


def exp_to_std(exp):
    exp = np.float64(exp)
    std = np.sqrt((exp + 1) / (exp + 2) ** 2 / (exp + 3))
    return std


def std_to_exp(std):
    std = np.float64(std)
    tmp = std.flatten() ** -2
    exp = [np.roots([1, 7, 16 - t, 12 - t]).real.max() for t in tmp]
    exp = np.float64(exp).reshape(std.shape)
    return exp


# -----------------------------------------------------------------------------
# Correlation between two power-function EMA profiles (Algorithm 3 in EDM2 paper).


def power_function_correlation(a_ofs, a_std, b_ofs, b_std):
    a_exp = std_to_exp(a_std)
    b_exp = std_to_exp(b_std)
    t_ratio = a_ofs / b_ofs
    t_exp = np.where(a_ofs < b_ofs, b_exp, -a_exp)
    t_max = np.maximum(a_ofs, b_ofs)
    num = (a_exp + 1) * (b_exp + 1) * t_ratio ** t_exp
    den = (a_exp + b_exp + 1) * t_max
    return num / den


def solve_posthoc_coefficients(in_ofs, in_std, out_ofs, out_std):  # => [in, out]
    """Solve coefficients for reconstructing out profiles from in profiles.

    Inputs are arrays/lists of:
    - in_ofs: training times (nimg) for each stored snapshot
    - in_std: s_rel for each stored snapshot
    - out_ofs: desired training time(s) (nimg), typically a scalar repeated
    - out_std: desired s_rel(s)
    Returns X with shape [len(in), len(out)] such that:
      theta_out[j] ~= sum_i X[i,j] * theta_in[i]
    """

    in_ofs, in_std = np.broadcast_arrays(in_ofs, in_std)
    out_ofs, out_std = np.broadcast_arrays(out_ofs, out_std)
    rv = lambda x: np.float64(x).reshape(-1, 1)
    cv = lambda x: np.float64(x).reshape(1, -1)
    A = power_function_correlation(rv(in_ofs), rv(in_std), cv(in_ofs), cv(in_std))
    B = power_function_correlation(rv(in_ofs), rv(in_std), cv(out_ofs), cv(out_std))
    X = np.linalg.solve(A, B)
    X = X / np.sum(X, axis=0)
    return X


# -----------------------------------------------------------------------------
# Online tracking of power-function EMA during training (Equation 127 update).


def power_function_beta(std, t_next, t_delta):
    """Compute per-update beta for a given s_rel at time t_next.

    std: s_rel
    t_next: current training time after the update (e.g., cur_nimg after increment)
    t_delta: step size in images (typically batch_size)
    """

    beta = (1 - t_delta / t_next) ** (std_to_exp(std) + 1)
    return beta


class PowerFunctionEMA:
    """Track multiple power-function EMA profiles as EMA networks.

    This maintains a set of EMA networks for different std (s_rel) values.
    """

    @torch.no_grad()
    def __init__(self, net, stds=(0.050, 0.100)):
        self.net = net
        self.stds = list(stds)
        self.emas = [copy.deepcopy(net).eval().requires_grad_(False) for _std in self.stds]

    @torch.no_grad()
    def reset(self):
        for ema in self.emas:
            for p_net, p_ema in zip(self.net.parameters(), ema.parameters()):
                p_ema.copy_(p_net)
            for b_net, b_ema in zip(self.net.buffers(), ema.buffers()):
                b_ema.copy_(b_net)

    @torch.no_grad()
    def update(self, *, cur_nimg, batch_size):
        # cur_nimg is expected to be the training time AFTER consuming the batch.
        if cur_nimg <= 0:
            return
        for std, ema in zip(self.stds, self.emas):
            beta = power_function_beta(std=std, t_next=cur_nimg, t_delta=batch_size)
            for p_net, p_ema in zip(self.net.parameters(), ema.parameters()):
                p_ema.lerp_(p_net, 1 - beta)
            for b_net, b_ema in zip(self.net.buffers(), ema.buffers()):
                b_ema.copy_(b_net)

    @torch.no_grad()
    def get(self):
        # Keep buffers in sync with the live net (mirrors edm2 behavior).
        for ema in self.emas:
            for b_net, b_ema in zip(self.net.buffers(), ema.buffers()):
                b_ema.copy_(b_net)
        return [(ema, f'-{std:.3f}') for std, ema in zip(self.stds, self.emas)]

    def state_dict(self):
        return dict(stds=self.stds, emas=[ema.state_dict() for ema in self.emas])

    def load_state_dict(self, state):
        self.stds = list(state['stds'])
        # If std set changed, rebuild EMA modules to match count.
        if len(self.emas) != len(self.stds):
            self.emas = [copy.deepcopy(self.net).eval().requires_grad_(False) for _ in self.stds]
        for ema, s_ema in zip(self.emas, state['emas']):
            ema.load_state_dict(s_ema)


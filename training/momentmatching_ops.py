from typing import Callable

import torch


@torch.no_grad()
def make_karras_sigmas(
    num_nodes: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    round_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Construct a monotonically descending Karras noise grid (length = num_nodes).

    Does not append 0; consumers treat 0 as a conceptual boundary.
    Each sigma is passed through ``round_fn`` (e.g. ``net.round_sigma``)
    to align with network rounding.

    Returns a 1-D tensor [sigma_0 > sigma_1 > ... > sigma_{num_nodes-1}].
    """
    assert num_nodes >= 1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    step_indices = torch.arange(num_nodes, dtype=torch.float64, device=device)
    sigmas = (
        sigma_max ** (1.0 / rho)
        + step_indices / max(num_nodes - 1, 1)
        * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    sigmas = round_fn(sigmas)
    return sigmas


def time_to_sigma(
    t: torch.Tensor,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> torch.Tensor:
    """Map continuous time t in [0, 1] to EDM sigma via Karras schedule.

    t=0 -> sigma_min, t=1 -> sigma_max.
    """
    t = t.to(torch.float64)
    inv_rho = 1.0 / rho
    sigma = ((1.0 - t) * sigma_min ** inv_rho + t * sigma_max ** inv_rho) ** rho
    return sigma


def sample_timesteps_mm(
    batch_size: int,
    k: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> dict:
    """Algorithm 2, Step 1: sample (sigma_s, sigma_t) for moment matching.

    s ~ Uniform(0, 1)        target time
    delta_t ~ U[0, 1/k]      random jump size
    t = min(s + delta_t, 1)   sampling time (noisier than s)

    Returns dict with float64 tensors of shape [N]:
        sigma_t, sigma_s, time_s, time_t
    """
    s = torch.rand(batch_size, device=device, dtype=torch.float64)
    delta_t = torch.rand(batch_size, device=device, dtype=torch.float64) / k
    t = torch.clamp(s + delta_t, max=1.0)

    sigma_s = time_to_sigma(s, sigma_min, sigma_max, rho)
    sigma_t = time_to_sigma(t, sigma_min, sigma_max, rho)

    return dict(
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        time_s=s,
        time_t=t,
    )


def sample_conditional_posterior(
    z_t: torch.Tensor,
    x_pred: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
) -> torch.Tensor:
    """Algorithm 2, Step 4: sample z_s ~ q(z_s | z_t, x_pred) in EDM space.

    EDM forward process: z = x + sigma * eps
    Conditional posterior (Bayes' rule on the EDM forward process):
        q(z_s | z_t, x) = N(mu, lambda^2 I)
        mu      = (sigma_t^2 - sigma_s^2) / sigma_t^2 * x  +  sigma_s^2 / sigma_t^2 * z_t
        lambda^2 = sigma_s^2 * (sigma_t^2 - sigma_s^2) / sigma_t^2

    All arithmetic in float64 for numerical stability.

    Args:
        z_t:     [N, C, H, W] noisy data at time t
        x_pred:  [N, C, H, W] clean-data prediction (x_tilde from student)
        sigma_t: [N] or broadcastable, noise level at t (> sigma_s)
        sigma_s: [N] or broadcastable, noise level at s
    Returns:
        z_s:     [N, C, H, W] sampled intermediate state
    """
    out_dtype = z_t.dtype
    z_t64 = z_t.to(torch.float64)
    x64 = x_pred.to(torch.float64)

    st = sigma_t.to(torch.float64).reshape(-1, 1, 1, 1)
    ss = sigma_s.to(torch.float64).reshape(-1, 1, 1, 1)

    st_sq = st * st
    ss_sq = ss * ss
    diff_sq = st_sq - ss_sq

    mu = (diff_sq / st_sq) * x64 + (ss_sq / st_sq) * z_t64
    lam = torch.sqrt(ss_sq * diff_sq / st_sq)

    eps = torch.randn_like(z_t64)
    z_s = mu + lam * eps
    return z_s.to(out_dtype)

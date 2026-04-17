# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist, persistence, misc
from training.momentmatching_ops import time_to_sigma, sample_conditional_posterior


#----------------------------------------------------------------------------
# Persistence import hook to make training snapshots robust to relative imports.
# Ensures checkpoints with relative imports can be loaded cleanly.

@persistence.import_hook
def _fix_relative_imports(meta: dnnlib.EasyDict) -> dnnlib.EasyDict:
    src = meta.module_src

    # training.momentmatching_ops: relative import -> absolute
    src = src.replace(
        "from .momentmatching_ops import ",
        "from training.momentmatching_ops import ",
    )

    # torch_utils.*: relative imports -> absolute
    src = src.replace(
        "from . import distributed as dist",
        "from torch_utils import distributed as dist",
    )
    src = src.replace(
        "from . import training_stats",
        "from torch_utils import training_stats",
    )
    src = src.replace(
        "from . import misc",
        "from torch_utils import misc",
    )

    # dnnlib.__init__: relative import -> absolute
    src = src.replace(
        "from .util import EasyDict, make_cache_dir_path",
        "from dnnlib.util import EasyDict, make_cache_dir_path",
    )

    meta.module_src = src
    return meta

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Ancestral sampler for moment-matching distilled models.
# Implements Algorithm 1 from "Multistep Distillation of Diffusion Models
# via Moment Matching" (Salimans & Ho, 2024).
#
# ───────────────────────────────────────────────────────────────────────
# Mathematical derivation  (VP notation → EDM space)
# ───────────────────────────────────────────────────────────────────────
#
# The Moment Matching paper works in the general VP diffusion framework
# where the forward process is:
#
#     q(z_t | x) = N(z_t ; alpha_t * x,  sigma_t^2 I)
#     with  sigma_t^2 = 1 - alpha_t^2          (VP constraint)
#
# The conditional posterior (reverse transition given the clean data x):
#
#     q(z_s | z_t, x) = N(z_s ; mu, lambda^2 I)
#     mu      = (alpha_s sigma_t^2 - alpha_t sigma_s^2) / sigma_t^2 * x
#             + alpha_t sigma_s^2 / sigma_t^2 * z_t
#     lambda^2 = sigma_s^2 (sigma_t^2 - sigma_s^2) / sigma_t^2
#     (Eq. 2 in the paper, with alpha_s/alpha_t simplification)
#
# EDM (Karras et al., 2022) uses the *specific* parameterisation where
# the forward process has *no signal scaling*:
#
#     z = x + sigma * epsilon          i.e.  alpha_t ≡ 1  for all t
#
# This is stated explicitly in EDM Table 1 row "s(t)=1" and Section 2.
# Substituting alpha_t = alpha_s = 1 into the VP posterior:
#
#     mu      = (sigma_t^2 - sigma_s^2) / sigma_t^2 * x
#             +  sigma_s^2             / sigma_t^2 * z_t
#
#     lambda^2 = sigma_s^2 * (sigma_t^2 - sigma_s^2) / sigma_t^2
#
# These are the equations implemented in sample_conditional_posterior()
# in training/momentmatching_ops.py and re-used here.
#
# ───────────────────────────────────────────────────────────────────────
# Algorithm 1 walkthrough  (adapted to EDM space)
# ───────────────────────────────────────────────────────────────────────
#
#   Inputs:  distilled model g_eta,  k steps,  sigma schedule {sigma_i}
#
#   1.  Build a monotonically decreasing sigma grid of length k+1:
#           sigma_0 > sigma_1 > ... > sigma_{k-1} > sigma_k
#       using the Karras schedule  sigma_i = time_to_sigma(i/k).
#       sigma_0 ≈ sigma_max,  sigma_k ≈ sigma_min.
#
#   2.  Draw initial noise:  z_{sigma_0} ~ N(0, sigma_0^2 I).
#
#   3.  For i = 0, …, k-1:
#       a. Evaluate the student:  x_hat = g_eta(z_{sigma_i}, sigma_i)
#       b. Sample the reverse transition:
#              z_{sigma_{i+1}} ~ q(z_{sigma_{i+1}} | z_{sigma_i}, x_hat)
#          using the EDM-space posterior above.
#       (When sigma_{i+1} = sigma_min ≈ 0, the posterior collapses to
#        a point mass at x_hat, so we just set z = x_hat.)
#
#   4.  Return z_{sigma_k} as the generated sample.
#
# Why this differs from the ODE sampler in edm_sampler():
#   - edm_sampler uses the score/probability-flow ODE and assumes the
#     network output is the *posterior mean* E[x|z].  It follows a
#     deterministic trajectory and can use higher-order solvers (Heun).
#   - The ancestral sampler treats the student output as a *sample*
#     x_hat ~ p(x|z) and draws from the true conditional posterior.
#     This is stochastic by construction and is mathematically consistent
#     with how the student was trained via moment matching.
# ───────────────────────────────────────────────────────────────────────

def ancestral_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=8, sigma_min=0.002, sigma_max=80, rho=7,
):
    """Ancestral sampler for moment-matching distilled models (Algorithm 1).

    Iterates over k = num_steps reverse transitions, each drawing from
    q(z_s | z_t, x_hat) where x_hat = net(z_t, sigma_t).
    """
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Build descending sigma grid: sigma_0 = sigma_max … sigma_k = sigma_min.
    step_indices = torch.arange(num_steps + 1, dtype=torch.float64, device=latents.device)
    t_grid = step_indices / num_steps                          # [0, 1/k, 2/k, ..., 1]
    sigma_grid = time_to_sigma(t_grid, sigma_min, sigma_max, rho)
    # time_to_sigma maps t=0 → sigma_min and t=1 → sigma_max, so reverse.
    sigma_grid = sigma_grid.flip(0)                            # [sigma_max, ..., sigma_min]
    sigma_grid = net.round_sigma(sigma_grid)

    # z_0 ~ N(0, sigma_max^2 I)
    x_next = latents.to(torch.float64) * sigma_grid[0]

    for i in range(num_steps):
        sig_t = sigma_grid[i]
        sig_s = sigma_grid[i + 1]

        # Student prediction: x_hat = g_eta(z_t, sigma_t)
        x_hat = net(x_next, sig_t, class_labels).to(torch.float64)

        if i == num_steps - 1:
            # Last step: sigma_s ≈ sigma_min ≈ 0; posterior collapses to x_hat.
            x_next = x_hat
        else:
            # Sample z_s ~ q(z_s | z_t, x_hat) using EDM-space posterior.
            sig_t_batch = sig_t.expand(x_next.shape[0])
            sig_s_batch = sig_s.expand(x_next.shape[0])
            x_next = sample_conditional_posterior(x_next, x_hat, sig_t_batch, sig_s_batch)

    return x_next

#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename (EMA snapshot); required if --state is not set', metavar='PATH|URL', type=str, required=False, default=None)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))
@click.option('--state', 'state_path',     help='Optional training-state-*.pt to load raw (non-EMA) weights',        metavar='PATH', type=str, default=None)
@click.option('--sampler',                 help='Sampler: edm (ODE), ablation, or ancestral (moment-matching)',     type=click.Choice(['edm', 'ablation', 'ancestral']), default=None)
def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, state_path, sampler, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network:
    # - If state_path is given, use the raw training net from training-state-*.pt (no EMA at all).
    # - Otherwise, fall back to EMA from snapshot for standard generation.
    if state_path is not None:
        dist.print0(f'Loading raw training weights from "{state_path}" (ignoring EMA)...')
        state = torch.load(state_path, map_location=torch.device('cpu'))
        if 'net' not in state:
            raise KeyError(f'"{state_path}" does not contain a \'net\' entry (expected raw training network).')
        net = state['net'].to(device)
        net.eval().requires_grad_(False)
    else:
        if network_pkl is None:
            raise click.ClickException("Missing required option '--network' when --state is not provided.")
        dist.print0(f'Loading network (EMA) from "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
            snapshot_data = pickle.load(f)
        if 'ema' not in snapshot_data:
            raise KeyError(f'"{network_pkl}" does not contain an EMA network (expected key \'ema\').')
        net = snapshot_data['ema'].to(device)
        net.eval().requires_grad_(False)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        if sampler == 'ancestral':
            anc_keys = {'num_steps', 'sigma_min', 'sigma_max', 'rho'}
            anc_kwargs = {k: v for k, v in sampler_kwargs.items() if k in anc_keys}
            sampler_fn = ancestral_sampler
            images = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, **anc_kwargs)
        elif sampler == 'ablation' or have_ablation_kwargs:
            images = ablation_sampler(net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)
        else:
            images = edm_sampler(net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

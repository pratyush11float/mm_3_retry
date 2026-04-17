# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import glob
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
import sys
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

# Validation hook.
from validation import maybe_validate, run_fid_validation

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    wandb_kwargs        = None,     # Options for Weights & Biases logging, None = disable.
    wandb_config        = None,     # Serializable run config to send to W&B (optional).
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    phema_stds          = None,     # Power-function EMA stds for post-hoc reconstruction (e.g. [0.05, 0.10]).
    phema_snapshot_ticks = None,    # How often to save PHEMA snapshots (ticks), None = use snapshot_ticks.
    lr_warmup_steps     = 1000,     # LR linear warmup duration in optimizer steps.
    lr_anneal           = True,     # Linearly anneal LR to zero after warmup.
    grad_clip           = 1.0,      # Gradient clipping max norm (0 = disable).
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    step_metrics_every  = 0,        # Emit per-step JSONL/W&B metrics every N steps (0 = disable).
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    validation_kwargs   = None,     # Validation configuration (PRD-04).
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    if hasattr(net, 'use_fp16') and not net.use_fp16:
        dist.print0('[OVERRIDE] Forcing use_fp16=True on student network')
        net.use_fp16 = True
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0(f'[EMA CONFIG] Validation EMA: halflife={ema_halflife_kimg} kimg, rampup_ratio={ema_rampup_ratio}')
    if ema_rampup_ratio is None or ema_rampup_ratio == 0:
        _fixed_beta = 0.5 ** (batch_size / max(ema_halflife_kimg * 1000, 1e-8))
        dist.print0(f'[EMA CONFIG]   No rampup → fixed beta={_fixed_beta:.6f} per step (halflife={ema_halflife_kimg*1000/batch_size:.1f} steps)')
    else:
        dist.print0(f'[EMA CONFIG]   Rampup active → effective halflife = min({ema_halflife_kimg} kimg, {ema_rampup_ratio}*cur_nimg)')
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    loss_fn._collect_step_metrics = False
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    # DDP configuration:
    # - broadcast_buffers=False: only GroupNorm (no running stats)
    # - gradient_as_bucket_view=True: avoids extra gradient copies during allreduce
    # - find_unused_parameters=False: all params used every iteration
    # - bucket_cap_mb=100: larger buckets → fewer allreduce calls over cross-node fabric
    _find_unused_env = os.environ.get('EDM_DDP_FIND_UNUSED_PARAMETERS', '').strip().lower()
    ddp_find_unused_parameters = False
    if _find_unused_env != '':
        ddp_find_unused_parameters = _find_unused_env not in ('0', 'false', 'no', 'off')
    _bucket_cap_mb = int(os.environ.get('EDM_DDP_BUCKET_CAP_MB', '100'))
    dist.print0(f'[DDP CONFIG] find_unused_parameters={ddp_find_unused_parameters} '
                f'gradient_as_bucket_view=True bucket_cap_mb={_bucket_cap_mb}')
    ddp = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[device],
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=False,
        find_unused_parameters=ddp_find_unused_parameters,
        bucket_cap_mb=_bucket_cap_mb,
    )
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Optional power-function EMA (EDM2 post-hoc EMA) for snapshotting.
    phema = None
    if phema_stds is not None and len(phema_stds) > 0:
        from training.phema import PowerFunctionEMA
        phema = PowerFunctionEMA(net=net, stds=tuple(phema_stds))
        phema.reset()
        _phema_tick = phema_snapshot_ticks if phema_snapshot_ticks is not None else snapshot_ticks
        dist.print0(f'[PHEMA] Enabled power-function EMA stds={list(phema_stds)}, snapshot every {_phema_tick} ticks')

    # Force fp16 on teacher network (matches student override above).
    if hasattr(loss_fn, 'teacher_net') and hasattr(loss_fn.teacher_net, 'use_fp16') and not loss_fn.teacher_net.use_fp16:
        dist.print0('[OVERRIDE] Forcing use_fp16=True on teacher network')
        loss_fn.teacher_net.use_fp16 = True

    # Seed student from teacher AFTER DDP wrapping (if distillation mode and shapes match).
    if hasattr(loss_fn, 'teacher_net'):
        try:
            teacher = loss_fn.teacher_net
            # Check if all params/buffers match in shape.
            def _shapes(mod):
                d = {}
                for n, p in mod.named_parameters():
                    d[n] = tuple(p.shape)
                for n, b in mod.named_buffers():
                    d[n] = tuple(b.shape)
                return d
            t_shapes = _shapes(teacher)
            s_shapes = _shapes(net)
            mismatches = []
            for name, tshape in t_shapes.items():
                sshape = s_shapes.get(name)
                if sshape is None or sshape != tshape:
                    mismatches.append((name, tshape, sshape))
            if len(mismatches) == 0:
                # All shapes match; copy weights to the underlying net module.
                misc.copy_params_and_buffers(src_module=teacher, dst_module=net, require_all=False)
                # Also update EMA to start from teacher weights.
                misc.copy_params_and_buffers(src_module=teacher, dst_module=ema, require_all=False)
                if dist.get_rank() == 0:
                    dist.print0('[INIT] Seeded student & EMA from teacher (all parameter/buffer shapes match).')
            else:
                if dist.get_rank() == 0:
                    dist.print0(f'[INIT] Not seeding from teacher because {len(mismatches)} tensors differ in shape.')
        except Exception as _e:
            if dist.get_rank() == 0:
                dist.print0(f'[INIT] Failed to seed from teacher: {_e}')
    

    # Moment Matching mode: set up auxiliary optimizer.
    is_mm_mode = hasattr(loss_fn, 'aux_net') and loss_fn.aux_net is not None
    optimizer_aux = None
    if is_mm_mode:
        aux_net = loss_fn.aux_net
        if hasattr(aux_net, 'use_fp16') and not aux_net.use_fp16:
            dist.print0('[OVERRIDE] Forcing use_fp16=True on auxiliary network')
            aux_net.use_fp16 = True
        optimizer_aux = dnnlib.util.construct_class_by_name(
            params=aux_net.parameters(), **optimizer_kwargs,
        )
        dist.print0(f'[MM INIT] Moment Matching mode enabled: alternating optimisation of student & auxiliary.')

    # Optional W&B initialization (async/threaded).
    wandb_run = None
    if wandb_kwargs is not None and wandb_kwargs.get('enabled', False) and dist.get_rank() == 0:
        try:
            import wandb as _wandb
            # Make W&B robust with our stdout redirection: avoid isatty uses.
            # Provide isatty() when stdout/stderr are our custom Logger without it.
            try:
                if not hasattr(sys.stdout, 'isatty'):
                    sys.stdout.isatty = lambda: False
                if not hasattr(sys.stderr, 'isatty'):
                    sys.stderr.isatty = lambda: False
            except Exception:
                pass
            init_kwargs = dict(
                project=wandb_kwargs.get('project', 'edm-momentmatch'),
                entity=wandb_kwargs.get('entity', None),
                name=wandb_kwargs.get('name', None),
                tags=wandb_kwargs.get('tags', None),
            )
            mode = wandb_kwargs.get('mode', 'online')
            if mode in ('offline', 'disabled'):
                init_kwargs['mode'] = mode
            # Use default settings so W&B can capture console logs.
            wandb_run = _wandb.init(**init_kwargs, config=wandb_config)
            # Also sync log.txt into the W&B run Files tab (live).
            try:
                _wandb.save(os.path.join(run_dir, 'log.txt'), policy='live')
            except Exception:
                pass
        except Exception as _e:
            dist.print0(f'[W&B] init failed: {_e}')
            wandb_run = None

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        if phema is not None and 'phema' in data:
            try:
                phema.load_state_dict(data['phema'])
                dist.print0('[PHEMA] Loaded power-function EMA state from training state.')
            except Exception as _e:
                dist.print0(f'[PHEMA] Failed to load PHEMA state: {_e}')
        if is_mm_mode and 'aux_net' in data:
            try:
                misc.copy_params_and_buffers(src_module=data['aux_net'], dst_module=loss_fn.aux_net, require_all=False)
                dist.print0('[MM] Loaded auxiliary model from training state.')
            except Exception as _e:
                dist.print0(f'[MM] Failed to load auxiliary model: {_e}')
        if is_mm_mode and optimizer_aux is not None and 'optimizer_aux_state' in data:
            try:
                optimizer_aux.load_state_dict(data['optimizer_aux_state'])
                dist.print0('[MM] Loaded auxiliary optimizer state from training state.')
            except Exception as _e:
                dist.print0(f'[MM] Failed to load auxiliary optimizer state: {_e}')
        del data # conserve memory

    # Broadcast run_dir to all ranks (rank 0 has it, others have None).
    if dist.get_rank() == 0:
        run_dir_str = run_dir if run_dir is not None else ''
    else:
        run_dir_str = ''
    # Use object list broadcast.
    run_dir_list = [run_dir_str]
    dist.ddp_debug(f'run_dir broadcast: before, run_dir_str="{run_dir_str}"')
    torch.distributed.broadcast_object_list(run_dir_list, src=0)
    run_dir = run_dir_list[0] if run_dir_list[0] else None
    dist.ddp_debug(f'run_dir broadcast: after, run_dir="{run_dir}"')

    # Pass run_dir to the loss function for debug logging.
    if run_dir is not None and hasattr(loss_fn, 'set_run_dir'):
        loss_fn.set_run_dir(run_dir)

    # One-time teacher validation (baseline) using ImageNet defaults if available.
    try:
        if (validation_kwargs is not None
            and validation_kwargs.get('enabled', True)
            and validation_kwargs.get('teacher', True)
            and hasattr(loss_fn, 'teacher_net')):
            # Rank 0 decides; broadcast to all.
            should_run_teacher = False
            if dist.get_rank() == 0:
                teacher_done_flag = os.path.join(run_dir, 'val_teacher.json')
                should_run_teacher = not os.path.isfile(teacher_done_flag)
            # Broadcast decision from rank 0.
            flag_tensor = torch.tensor([1 if should_run_teacher else 0], dtype=torch.int64, device=device)
            dist.ddp_debug(f'teacher_flag broadcast: before, val={int(flag_tensor.item())}')
            torch.distributed.broadcast(flag_tensor, src=0)
            should_run_teacher = bool(flag_tensor.item())
            dist.ddp_debug(f'teacher_flag broadcast: after, val={int(flag_tensor.item())}')
            # Global sync so either all enter validation together or none do.
            torch.distributed.barrier()
            if should_run_teacher:
                # Use the same teacher object that the loss function uses.
                teacher_net = loss_fn.teacher_net.eval().requires_grad_(False).to(device)
                # Teacher sampler defaults per README ImageNet.
                # IMPORTANT: Do NOT override sigma_min/sigma_max here — edm_sampler
                # has good defaults (0.002, 80) and clamps against net.sigma_min/max.
                # Passing net.sigma_min/max (=0, inf) would break the grid and cause NaNs.
                teacher_sampler = dict(
                    kind='edm',
                    num_steps=256,
                    rho=7.0,
                    S_churn=40, S_min=0.05, S_max=50.0, S_noise=1.003,
                )
                dist.print0('[VAL] Running one-time teacher validation (ImageNet defaults)...')
                teacher_dump_dir = os.path.join(run_dir, 'teacher_samples') if dist.get_rank() == 0 else None
                result = run_fid_validation(
                    teacher_net,
                    run_dir=run_dir,
                    dataset_kwargs=dataset_kwargs,
                    num_images=int(validation_kwargs.get('num_images', 50000)),
                    batch=int(validation_kwargs.get('batch', 64)),
                    seed=int(validation_kwargs.get('seed', 0)),
                    sampler=teacher_sampler,
                    labels='auto',
                    ref=validation_kwargs.get('ref', None),
                    ref_data=validation_kwargs.get('ref_data', None),
                    dump_images_dir=teacher_dump_dir,
                    overwrite=False,
                    step_kimg=None,
                    wandb_run=wandb_run,
                )
                if dist.get_rank() == 0:
                    payload = dict(fid=result.get('fid'), sampler=teacher_sampler)
                    with open(teacher_done_flag, 'wt') as f:
                        json.dump(payload, f)
    except Exception as _e:
        dist.print0(f'[VAL] teacher validation failed: {_e}')

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    step_stats_jsonl = None
    total_steps = int(total_kimg * 1000 / batch_size)
    cur_step = int(resume_kimg * 1000 / batch_size) if resume_kimg > 0 else 0
    ema_updates = 0
    last_loss_scalar = None
    step_metrics_every = max(int(step_metrics_every or 0), 0)
    while True:
        if os.environ.get('CD_DDP_DEBUG'):
            print(f'[RANK {dist.get_rank()}] loop iter start: cur_nimg={cur_nimg}', flush=True)

        # Make teacher annealing resume-aware by exposing global kimg to the loss.
        try:
            if hasattr(loss_fn, 'set_global_kimg'):
                loss_fn.set_global_kimg(cur_nimg / 1e3)
        except Exception:
            pass

        # Accumulate gradients.
        # Tell loss_fn whether to build _last_step_metrics this step.
        _will_collect = step_metrics_every > 0 and (cur_step % step_metrics_every == 0)
        if hasattr(loss_fn, '_collect_step_metrics'):
            loss_fn._collect_step_metrics = _will_collect

        # Determine even/odd for MM alternation.
        mm_is_even = is_mm_mode and (cur_step % 2 == 0)
        if is_mm_mode and hasattr(loss_fn, 'set_step_n'):
            loss_fn.set_step_n(cur_step)

        # Compute LR multiplier: linear warmup then linear anneal to zero.
        if cur_step < lr_warmup_steps:
            lr_mult = cur_step / max(lr_warmup_steps, 1)
        elif lr_anneal and total_steps > lr_warmup_steps:
            lr_mult = max(0.0, 1.0 - (cur_step - lr_warmup_steps) / (total_steps - lr_warmup_steps))
        else:
            lr_mult = 1.0
        current_lr_value = optimizer_kwargs['lr'] * lr_mult

        if os.environ.get('CD_DDP_DEBUG'):
            print(f'[RANK {dist.get_rank()}] zero_grad', flush=True)

        if mm_is_even:
            # MM even step: update auxiliary model phi.
            # Student runs under no_grad inside loss_fn; pass raw net (no DDP).
            optimizer_aux.zero_grad(set_to_none=True)
            for round_idx in range(num_accumulation_rounds):
                images, labels = next(dataset_iterator)
                images = images.to(device, non_blocking=True).to(torch.float32) / 127.5 - 1
                labels = labels.to(device, non_blocking=True)
                loss = loss_fn(net=net, images=images, labels=labels, augment_pipe=augment_pipe)
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()
            # Manual gradient allreduce for auxiliary across ranks.
            for p in loss_fn.aux_net.parameters():
                if p.grad is not None:
                    torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)
                    torch.distributed.all_reduce(p.grad)
                    p.grad.div_(dist.get_world_size())
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(loss_fn.aux_net.parameters(), max_norm=grad_clip)
            for g in optimizer_aux.param_groups:
                g['lr'] = current_lr_value
            optimizer_aux.step()
        else:
            # Standard path (also used for MM odd steps).
            optimizer.zero_grad(set_to_none=True)
            for round_idx in range(num_accumulation_rounds):
                if os.environ.get('CD_DDP_DEBUG'):
                    print(f'[RANK {dist.get_rank()}] round {round_idx}: fetching batch', flush=True)
                with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                    images, labels = next(dataset_iterator)
                    if os.environ.get('CD_DDP_DEBUG'):
                        print(f'[RANK {dist.get_rank()}] round {round_idx}: batch fetched, moving to device', flush=True)
                    images = images.to(device, non_blocking=True).to(torch.float32) / 127.5 - 1
                    labels = labels.to(device, non_blocking=True)
                    if os.environ.get('CD_DDP_DEBUG'):
                        print(f'[RANK {dist.get_rank()}] round {round_idx}: calling loss_fn', flush=True)
                    loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                    if os.environ.get('CD_DDP_DEBUG'):
                        print(f'[RANK {dist.get_rank()}] round {round_idx}: loss computed, calling backward', flush=True)
                    training_stats.report('Loss/loss', loss)
                    loss.sum().mul(loss_scaling / batch_gpu_total).backward()
                    if os.environ.get('CD_DDP_DEBUG'):
                        print(f'[RANK {dist.get_rank()}] round {round_idx}: backward done', flush=True)

        last_loss_scalar = float(loss.mean().detach().cpu().item())

        # Update weights.
        collect_step_metrics = step_metrics_every > 0 and (cur_step % step_metrics_every == 0)
        if not mm_is_even:
            for g in optimizer.param_groups:
                g['lr'] = current_lr_value
        current_lr = optimizer.param_groups[0]['lr']
        grad_global_norm = None
        param_global_norm = None
        true_update_over_param = None
        # Sanitize gradients always; only compute expensive global norms when step metrics are enabled.
        if not mm_is_even:
            if collect_step_metrics:
                total_grad_sq = torch.zeros([], device=device)
                total_param_sq = torch.zeros([], device=device)
                for param in net.parameters():
                    param_norm = param.detach().float().norm(2)
                    total_param_sq = total_param_sq + param_norm * param_norm
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                        grad_norm = param.grad.detach().float().norm(2)
                        total_grad_sq = total_grad_sq + grad_norm * grad_norm
                grad_global_norm = torch.sqrt(total_grad_sq)
                param_global_norm = torch.sqrt(total_param_sq)
                true_update_over_param = (current_lr * grad_global_norm) / torch.clamp(param_global_norm, min=1e-12)
                training_stats.report('Grad/global_norm', grad_global_norm)
                training_stats.report('Grad/param_norm', param_global_norm)
                training_stats.report('Grad/update_over_param', true_update_over_param)
            else:
                for param in net.parameters():
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
            optimizer.step()
        # Per-optimizer-step diagnostics (rank 0 only, written to step_stats.jsonl and W&B).
        if collect_step_metrics and dist.get_rank() == 0:
            if step_stats_jsonl is None and run_dir is not None:
                step_stats_path = os.path.join(run_dir, 'step_stats.jsonl')
                step_stats_jsonl = open(step_stats_path, 'at')
            # Build a per-step metrics payload from loss + optimizer state.
            # Compute validation EMA beta for logging (mirrors the update logic below).
            _ema_hl_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None:
                _ema_hl_nimg = min(_ema_hl_nimg, cur_nimg * ema_rampup_ratio)
            _ema_beta_log = 0.5 ** (batch_size / max(_ema_hl_nimg, 1e-8))
            step_record = {
                'step': cur_step,
                'nimg': int(cur_nimg),
                'kimg': float(cur_nimg / 1e3),
                'tick': int(cur_tick),
                'ema_beta': float(_ema_beta_log),
                'lr': float(current_lr),
                'grad_global_norm': float(grad_global_norm.detach().cpu()) if grad_global_norm is not None else None,
                'param_global_norm': float(param_global_norm.detach().cpu()) if param_global_norm is not None else None,
                'update_over_param': float(true_update_over_param.detach().cpu()) if true_update_over_param is not None else None,
                'loss': float(last_loss_scalar) if last_loss_scalar is not None else None,
            }
            if hasattr(loss_fn, '_last_step_metrics') and isinstance(getattr(loss_fn, '_last_step_metrics'), dict):
                step_record.update(loss_fn._last_step_metrics)
            # Write to per-step JSONL if available.
            if step_stats_jsonl is not None:
                step_stats_jsonl.write(json.dumps(step_record) + '\n')
                step_stats_jsonl.flush()
            # Also log per-step metrics to W&B (no tick aggregation).
            if wandb_run is not None:
                try:
                    import wandb as _wandb
                    wandb_payload = {
                        'opt_step': step_record['step'],
                        'kimg': step_record['kimg'],
                        'EMA/val_ema_beta': step_record['ema_beta'],
                    }
                    if step_record['grad_global_norm'] is not None:
                        wandb_payload['Grad/global_norm'] = step_record['grad_global_norm']
                    if step_record['param_global_norm'] is not None:
                        wandb_payload['Grad/param_norm'] = step_record['param_global_norm']
                    if step_record['update_over_param'] is not None:
                        wandb_payload['Grad/update_over_param'] = step_record['update_over_param']
                    # Per-step loss (separate from tick-level summary loss).
                    if step_record['loss'] is not None:
                        wandb_payload['Loss/loss_step'] = step_record['loss']
                    _wandb.log(wandb_payload, commit=True)
                except Exception as _e:
                    dist.print0(f'[W&B] per-step log failed: {_e}')
        cur_step += 1

        # Update EMA (validation/snapshot EMA).
        # In MM mode, only update EMA on odd steps (when student is updated).
        if not mm_is_even:
            ema_halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
            for p_ema, p_net in zip(ema.parameters(), net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
            ema_updates += 1

            # Update power-function EMA profiles (post-hoc EMA basis).
            if phema is not None:
                phema.update(cur_nimg=(cur_nimg + batch_size), batch_size=batch_size)

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        if os.environ.get('CD_DDP_DEBUG'):
            print(f'[RANK {dist.get_rank()}] tick end: gathering stats', flush=True)
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        if os.environ.get('CD_DDP_DEBUG'):
            print(f'[RANK {dist.get_rank()}] tick end: printing status', flush=True)
        dist.print0(' '.join(fields))
        if os.environ.get('CD_DDP_DEBUG'):
            print(f'[RANK {dist.get_rank()}] tick end: status printed', flush=True)

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
            # Optional W&B logging once per tick (async).
            if wandb_run is not None:
                try:
                    # Tick-level progress-only logging; detailed metrics are logged per optimizer step.
                    log_dict = dict(
                        progress_kimg=cur_nimg / 1e3,
                        tick=cur_tick,
                        loss=last_loss_scalar if last_loss_scalar is not None else None,
                        ema_updates=ema_updates,
                    )
                    import wandb as _wandb
                    _wandb.log(log_dict, commit=True)
                except Exception as _e:
                    dist.print0(f'[W&B] log failed: {_e}')
        dist.update_progress(cur_nimg // 1000, total_kimg)
        
        # Synchronize all ranks after rank-0-only logging to prevent desync before collective ops.
        torch.distributed.barrier()

        # Built-in validation hook (runs on schedule; blocks training).
        if os.environ.get('CD_DDP_DEBUG'):
            print(f'[RANK {dist.get_rank()}] before maybe_validate', flush=True)
        try:
            cur_kimg = cur_nimg // 1000
            maybe_validate(
                step_tick=cur_tick,
                step_kimg=int(cur_kimg),
                net_ema=ema,
                run_dir=run_dir,
                dataset_kwargs=dataset_kwargs,
                validation_kwargs=validation_kwargs,
                wandb_run=wandb_run,
            )
        except Exception as _e:
            dist.print0(f'[VAL] validation failed: {_e}')
        if os.environ.get('CD_DDP_DEBUG'):
            print(f'[RANK {dist.get_rank()}] after maybe_validate', flush=True)

        # Save network snapshot (after validation on this tick).
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            if os.environ.get('CD_DDP_DEBUG'):
                print(f'[RANK {dist.get_rank()}] snapshot: starting', flush=True)
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            if os.environ.get('CD_DDP_DEBUG'):
                print(f'[RANK {dist.get_rank()}] snapshot: processing {len(data)} items', flush=True)
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    if os.environ.get('CD_DDP_DEBUG'):
                        print(f'[RANK {dist.get_rank()}] snapshot: deepcopy {key}', flush=True)
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if os.environ.get('CD_DDP_DEBUG'):
                        print(f'[RANK {dist.get_rank()}] snapshot: check_ddp_consistency {key}', flush=True)
                    misc.check_ddp_consistency(value)
                    if os.environ.get('CD_DDP_DEBUG'):
                        print(f'[RANK {dist.get_rank()}] snapshot: move {key} to CPU', flush=True)
                    data[key] = value.cpu()
                    if os.environ.get('CD_DDP_DEBUG'):
                        print(f'[RANK {dist.get_rank()}] snapshot: {key} done', flush=True)
                del value  # conserve memory
            if os.environ.get('CD_DDP_DEBUG'):
                print(f'[RANK {dist.get_rank()}] snapshot: writing to disk (rank 0 only)', flush=True)
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            # CRITICAL: All ranks must wait for rank 0 to finish writing before continuing.
            # Without this barrier, other ranks will start the next training iteration while
            # rank 0 is still writing, causing NCCL timeout when they try to synchronize.
            torch.distributed.barrier()
            if os.environ.get('CD_DDP_DEBUG'):
                print(f'[RANK {dist.get_rank()}] snapshot: done', flush=True)
            del data  # conserve memory

        # Save power-function EMA snapshots (one pickle per std) for post-hoc reconstruction.
        # Uses its own cadence (phema_snapshot_ticks), independent of main snapshots.
        _phema_tick = phema_snapshot_ticks if phema_snapshot_ticks is not None else snapshot_ticks
        if phema is not None and (_phema_tick is not None) and (done or cur_tick % _phema_tick == 0):
            phema_list = phema.get()  # [(ema_net, suffix_str)]
            base = dict(loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            base_cpu = {}
            for key, value in base.items():
                if isinstance(value, torch.nn.Module):
                    v = copy.deepcopy(value).eval().requires_grad_(False)
                    try:
                        misc.check_ddp_consistency(v)
                    except Exception:
                        pass
                    base_cpu[key] = v.cpu()
                    del v
                else:
                    base_cpu[key] = value
                del value
            for ema_net, ema_suffix in phema_list:
                data_phema = dict(base_cpu, ema=copy.deepcopy(ema_net).eval().requires_grad_(False).cpu(), nimg=cur_nimg)
                if dist.get_rank() == 0:
                    std_str = ema_suffix.lstrip('-')  # e.g. "0.050"
                    fname = f'network-snapshot-{cur_nimg//1000:06d}-{std_str}.pkl'
                    with open(os.path.join(run_dir, fname), 'wb') as f:
                        pickle.dump(data_phema, f)
                del data_phema
            torch.distributed.barrier()
            del base_cpu, base, phema_list

        # Save full dump of the training state (after validation and snapshot).
        # NOTE: Only rank 0 writes the file, but ALL ranks must wait before starting the
        # next iteration to avoid some ranks entering the next backward() while rank 0
        # is still busy with torch.save(), which would otherwise cause NCCL allreduce
        # timeouts (some ranks enqueuing gradient allreduces while others haven't yet).
        need_state_dump = (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0
        if need_state_dump and dist.get_rank() == 0:
            if os.environ.get('CD_DDP_DEBUG'):
                print(f'[RANK {dist.get_rank()}] state_dump: starting', flush=True)
            state_dict = dict(net=net, optimizer_state=optimizer.state_dict())
            if phema is not None:
                state_dict['phema'] = phema.state_dict()
            if is_mm_mode:
                state_dict['aux_net'] = loss_fn.aux_net
                state_dict['optimizer_aux_state'] = optimizer_aux.state_dict()
            state_path = os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt')
            _max_save_attempts = 3
            for _attempt in range(1, _max_save_attempts + 1):
                try:
                    torch.save(state_dict, state_path)
                    break
                except Exception as _save_err:
                    # Remove any partial file left behind by the failed write.
                    try:
                        os.remove(state_path)
                    except OSError:
                        pass
                    if _attempt < _max_save_attempts:
                        dist.print0(f'[STATE DUMP] Save attempt {_attempt}/{_max_save_attempts} failed ({_save_err}); retrying in 30s...')
                        time.sleep(30)
                    else:
                        dist.print0(f'[STATE DUMP] All {_max_save_attempts} save attempts failed; skipping state dump at kimg={cur_nimg//1000}.')
            # Keep the 2 most recent state files plus the one corresponding to the
            # best validation FID seen so far. Delete everything else.
            best_fid_state = None
            metrics_val_path = os.path.join(run_dir, 'metrics-val.jsonl')
            if os.path.exists(metrics_val_path):
                try:
                    best_entry = min(
                        (json.loads(line) for line in open(metrics_val_path) if line.strip()),
                        key=lambda x: x.get('fid', float('inf')),
                        default=None,
                    )
                    if best_entry is not None and best_entry.get('kimg') is not None:
                        best_fid_state = os.path.join(run_dir, f'training-state-{int(best_entry["kimg"]):06d}.pt')
                except Exception:
                    pass
            all_states = sorted(glob.glob(os.path.join(run_dir, 'training-state-*.pt')))
            protected = set(all_states[-2:])
            if best_fid_state is not None:
                protected.add(best_fid_state)
            for _old_state_path in all_states:
                if _old_state_path not in protected:
                    try:
                        os.remove(_old_state_path)
                    except OSError:
                        pass
            if os.environ.get('CD_DDP_DEBUG'):
                print(f'[RANK {dist.get_rank()}] state_dump: done', flush=True)
        # All ranks participate in this barrier exactly on the sWhen you speak to it and it types whatever you're saying, but it's crazy accurate like that is for likeame ticks where we dump
        # state, ensuring no one starts the next iteration early relative to rank 0.
        if need_state_dump:
            torch.distributed.barrier()

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')
    # Close W&B run.
    try:
        if wandb_run is not None and dist.get_rank() == 0:
            import wandb as _wandb
            _wandb.finish()
    except Exception:
        pass

#----------------------------------------------------------------------------

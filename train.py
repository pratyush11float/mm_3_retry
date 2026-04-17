# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch
import dnnlib
import pickle
import copy
from torch_utils import distributed as dist
from training import training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

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

def parse_float_list(s):
    if s is None:
        return None
    if isinstance(s, list):
        return [float(x) for x in s]
    s = str(s).strip()
    if s == '' or s.lower() in ('none', 'null', 'off', '0'):
        return None
    return [float(x.strip()) for x in s.split(',') if x.strip() != '']

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm',       type=click.Choice(['vp', 've', 'edm']), default='edm', show_default=True)

# Moment Matching Distillation options.
@click.option('--teacher',       help='Path/URL to pre-trained teacher (EDM-precond UNet)', metavar='PKL|URL', type=str)
@click.option('--S',             'S', help='Student step count (k)', metavar='INT', type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--rho',           help='Karras rho exponent', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=7.0, show_default=True)
@click.option('--sigma_min',     help='Minimum sigma', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',     help='Maximum sigma', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=80.0, show_default=True)
@click.option('--sync_dropout/--no_sync_dropout', help='Synchronize CUDA RNG for dropout between g_phi and g_theta', default=True, show_default=True)
@click.option('--momentmatch',   help='Enable Moment Matching Distillation (Algorithm 2)', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option(
    '--mm_weight_mode',
    help='Moment matching weight mode',
    metavar='edm|vlike|flat|snr|snr+1|karras|sqrt_karras|truncated-snr|uniform',
    type=click.Choice(['edm', 'vlike', 'flat', 'snr', 'snr+1', 'karras', 'sqrt_karras', 'truncated-snr', 'uniform']),
    default='edm',
    show_default=True,
)
@click.option('--momentmatch_algo', type=click.Choice(['2', '3']), default='3', show_default=True)
@click.option('--mm_precond', type=click.Choice(['identity', 'adam']), default='identity', show_default=True)
@click.option('--teacher_state_dump', metavar='PT', type=str, default=None)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=2e-6, show_default=True)
@click.option('--beta1',         help='Adam beta1', metavar='FLOAT',                                type=click.FloatRange(min=0, max=1), default=0.0, show_default=True)
@click.option('--beta2',         help='Adam beta2', metavar='FLOAT',                                type=click.FloatRange(min=0, max=1), default=0.99, show_default=True)
@click.option('--adam_eps',      help='Adam epsilon', metavar='FLOAT',                              type=float, default=1e-12, show_default=True)
@click.option('--lr_warmup_steps', help='LR linear warmup steps', metavar='INT',                    type=click.IntRange(min=0), default=1000, show_default=True)
@click.option('--lr_anneal',    help='Linearly anneal LR to zero after warmup', metavar='BOOL',     type=bool, default=True, show_default=True)
@click.option('--grad_clip',    help='Gradient clipping max norm (0=disable)', metavar='FLOAT',     type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--ema_rampup',    help='EMA rampup ratio (0=disable rampup, Song uses 0)', metavar='FLOAT', type=click.FloatRange(min=0), default=0.05, show_default=True)
@click.option('--phema',         help='Power-function EMA stds for post-hoc reconstruction (comma list, e.g. 0.05,0.10; empty=off)', metavar='LIST', type=str, default='', show_default=True)
@click.option('--phema_snap',    help='PHEMA snapshot interval in ticks (default: same as --snap)', metavar='TICKS', type=click.IntRange(min=1), default=None)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.00, show_default=True)
@click.option('--dout_resolutions', help='Apply dropout only at these resolutions (comma-separated, e.g. 16,8). None=all.', metavar='LIST', type=str, default=None)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)
# Weights & Biases (optional).
@click.option('--wandb',         help='Enable Weights & Biases logging', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--wandb_project', help='W&B project name', metavar='STR',                            type=str, default='edm-momentmatch', show_default=True)
@click.option('--wandb_entity',  help='W&B entity (team/user)', metavar='STR',                      type=str)
@click.option('--wandb_run',     help='W&B run name', metavar='STR',                                type=str)
@click.option('--wandb_tags',    help='W&B tags (comma-separated)', metavar='STR',                  type=str)
@click.option('--wandb_mode',    help='W&B mode: online|offline|disabled', metavar='STR',           type=click.Choice(['online','offline','disabled']), default='online', show_default=True)

# Validation (PRD-04).
@click.option('--val',           help='Enable periodic validation (FID)', metavar='BOOL',            type=bool, default=True, show_default=True)
@click.option('--val_ref',       help='FID reference stats (.npz or URL)', metavar='NPZ|URL',        type=str)
@click.option('--val_ref_data',  help='Dataset path to compute reference', metavar='PATH',           type=str)
@click.option('--val_every',     help='Validate every N ticks (default=snap)', metavar='TICKS',      type=click.IntRange(min=1))
@click.option('--val_num',       help='Number of images for validation', metavar='INT',              type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--val_seed',      help='Validation base seed', metavar='INT',                         type=int, default=0, show_default=True)
@click.option('--val_batch',     help='Validation batch size per GPU', metavar='INT',                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--val_steps',     help='Validation sampler steps', metavar='INT',                     type=click.IntRange(min=1), default=16, show_default=True)
@click.option('--val_sampler',   help='Sampler kind', metavar='edm|ablate|ancestral',                 type=click.Choice(['edm','ablate','ancestral']), default='edm', show_default=True)
@click.option('--val_label',     help='Label mode', metavar='auto|uniform|dataset|fixed:K',         type=str, default='auto', show_default=True)
@click.option('--val_dump_images_dir', help='Optional: dump validation images', metavar='DIR',       type=str)
@click.option('--val_overwrite', help='Overwrite existing val_{kimg}.json', metavar='BOOL',          type=bool, default=False, show_default=True)
@click.option('--val_at_start',  help='Run validation at start (tick 0)', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--val_teacher',   help='Run one-time teacher baseline validation at start', metavar='BOOL', type=bool, default=True, show_default=True)

def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)

    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[opts.beta1, opts.beta2], eps=opts.adam_eps)

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        # Debug: dataset label diagnostics.
        dist.print0(f'Dataset has_labels={dataset_obj.has_labels}, label_dim={getattr(dataset_obj, "label_dim", None)}')
        # If possible, peek into dataset.json to count unique labels.
        try:
            import zipfile, json as _json
            if isinstance(c.dataset_kwargs.path, str) and c.dataset_kwargs.path.endswith('.zip') and os.path.isfile(c.dataset_kwargs.path):
                with zipfile.ZipFile(c.dataset_kwargs.path, 'r') as zf:
                    if 'dataset.json' in zf.namelist():
                        with zf.open('dataset.json') as f:
                            meta = _json.load(f)
                            labels = meta.get('labels')
                            if labels is not None:
                                uniq = sorted({int(lbl) for _, lbl in labels if lbl is not None})
                                dist.print0(f'Dataset.json unique label count={len(uniq)}, min={uniq[0] if uniq else None}, max={uniq[-1] if uniq else None}')
        except Exception as _e:
            dist.print0(f'Dataset label diagnostics skipped: {_e}')
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    # Preconditioning & loss function.
    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'training.networks.VPPrecond'
        c.loss_kwargs.class_name = 'training.loss.VPLoss'
    elif opts.precond == 've':
        c.network_kwargs.class_name = 'training.networks.VEPrecond'
        c.loss_kwargs.class_name = 'training.loss.VELoss'
    else:
        assert opts.precond == 'edm'
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'

    # Moment Matching Distillation wiring (overrides loss when enabled).
    if opts.momentmatch:
        if opts.precond != 'edm':
            raise click.ClickException('--momentmatch=True requires --precond=edm')
        if not opts.teacher:
            raise click.ClickException('--momentmatch=True requires --teacher=PKL|URL')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with dnnlib.util.open_url(opts.teacher, verbose=(dist.get_rank() == 0)) as f:
            teacher_data = pickle.load(f)

        if isinstance(teacher_data, dict) and ('ema' in teacher_data or 'net' in teacher_data):
            teacher = teacher_data['ema'] if 'ema' in teacher_data else teacher_data['net']
        else:
            teacher = teacher_data

        teacher = teacher.eval().to(device)
        
        # ----------------------------
        # Algorithm 2: create aux_net
        # ----------------------------

        if opts.momentmatch_algo == '2':
            aux_net = copy.deepcopy(teacher).train().requires_grad_(True).to(device)
            c.loss_kwargs.class_name = 'training.loss_mm.EDMMomentMatchLoss'
            c.loss_kwargs.update(
                teacher_net=teacher,
                aux_net=aux_net,
                k=opts.S,
                sigma_min=opts.sigma_min,
                sigma_max=opts.sigma_max,
                rho=opts.rho,
                weight_mode=opts.mm_weight_mode,
                sync_dropout=opts.sync_dropout,
            )
        
        # ----------------------------
        # Algorithm 3: DO NOT create aux_net
        # ----------------------------    
        else:
            c.loss_kwargs.class_name = 'training.loss_mm.EDMInstantMomentMatchLoss'
            c.loss_kwargs.update(
                teacher_net=teacher,
                k=opts.S,
                sigma_min=opts.sigma_min,
                sigma_max=opts.sigma_max,
                rho=opts.rho,
                weight_mode=opts.mm_weight_mode,
                precond_mode=opts.mm_precond,
                teacher_state_dump=opts.teacher_state_dump,
            )
        c.teacher = opts.teacher

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    # Augmentation policy:
    # For distillation, disable augmentation to match paper setup
    # and keep student identical to teacher (avoid augment_dim mismatch).
    if (not opts.momentmatch) and opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        # Only set augment_dim when we actually use AugmentPipe.
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)
    if opts.dout_resolutions is not None:
        c.network_kwargs.dout_resolutions = [int(x.strip()) for x in opts.dout_resolutions.split(',') if x.strip()]

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.ema_rampup_ratio = opts.ema_rampup if opts.ema_rampup > 0 else None
    c.phema_stds = parse_float_list(opts.phema)
    c.phema_snapshot_ticks = opts.phema_snap
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)
    c.update(lr_warmup_steps=opts.lr_warmup_steps, lr_anneal=opts.lr_anneal, grad_clip=opts.grad_clip)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        dist.ddp_debug(f'seed broadcast: before, local seed={int(seed)}')
        torch.distributed.broadcast(seed, src=0)
        dist.ddp_debug(f'seed broadcast: after, global seed={int(seed)}')
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.momentmatch:
        desc += f'-mmK{opts.S}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    # Sanitize non-serializable fields for printing.
    c_print = copy.deepcopy(c)
    if 'loss_kwargs' in c_print and isinstance(c_print.loss_kwargs, dnnlib.EasyDict):
        if 'teacher_net' in c_print.loss_kwargs:
            c_print.loss_kwargs.teacher_net = 'FROZEN_TEACHER'
        if 'aux_net' in c_print.loss_kwargs:
            c_print.loss_kwargs.aux_net = 'AUX_MODEL'
    dist.print0(json.dumps(c_print, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Moment Matching:         {opts.momentmatch}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            c_save = copy.deepcopy(c)
            if 'loss_kwargs' in c_save and isinstance(c_save.loss_kwargs, dnnlib.EasyDict):
                if 'teacher_net' in c_save.loss_kwargs:
                    c_save.loss_kwargs.teacher_net = 'FROZEN_TEACHER'
                if 'aux_net' in c_save.loss_kwargs:
                    c_save.loss_kwargs.aux_net = 'AUX_MODEL'
            json.dump(c_save, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Prepare W&B integration (optional).
    if opts.wandb:
        # Build a JSON-serializable copy of options to store as config.
        try:
            wandb_config = json.loads(json.dumps(c))
        except Exception:
            # As a fallback, store a minimal config.
            wandb_config = dict(
                dataset_kwargs=c.get('dataset_kwargs', {}),
                network_kwargs=dict(
                    class_name=c['network_kwargs'].get('class_name'),
                    model_type=c['network_kwargs'].get('model_type'),
                    model_channels=c['network_kwargs'].get('model_channels'),
                    channel_mult=c['network_kwargs'].get('channel_mult'),
                    use_fp16=c['network_kwargs'].get('use_fp16'),
                ),
                loss_kwargs=c.get('loss_kwargs', {}),
                optimizer_kwargs=c.get('optimizer_kwargs', {}),
                batch_size=c.get('batch_size'),
                total_kimg=c.get('total_kimg'),
                ema_halflife_kimg=c.get('ema_halflife_kimg'),
                seed=c.get('seed'),
            )
        # Pass wandb args to training loop.
        tags = None
        if opts.wandb_tags:
            tags = [t.strip() for t in opts.wandb_tags.split(',') if t.strip()]
        c.wandb_kwargs = dict(
            enabled=True,
            project=opts.wandb_project,
            entity=opts.wandb_entity,
            name=opts.wandb_run,
            tags=tags,
            mode=opts.wandb_mode,
        )
        c.wandb_config = wandb_config
    else:
        c.wandb_kwargs = None
        c.wandb_config = None

    # Validation configuration (PRD-04).
    # Teacher baseline in training_loop.py uses the stochastic ImageNet defaults
    # for its one-time FID with the original EDM Heun sampler.
    val_sampler_kind = opts.val_sampler
    if val_sampler_kind == 'ancestral':
        val_sampler_cfg = dict(
            kind='ancestral',
            num_steps=opts.val_steps,
            sigma_min=opts.sigma_min,
            sigma_max=opts.sigma_max,
            rho=opts.rho,
        )
    elif val_sampler_kind == 'ablate':
        val_sampler_cfg = dict(
            kind='ablate',
            num_steps=opts.val_steps,
            sigma_min=opts.sigma_min,
            sigma_max=opts.sigma_max,
            rho=opts.rho,
            solver='euler',
            discretization='edm',
            schedule='linear',
            scaling='none',
            S_churn=0.0, S_min=0.0, S_max=0.0, S_noise=1.0,
        )
    else:
        val_sampler_cfg = dict(
            kind='edm',
            num_steps=opts.val_steps,
            sigma_min=opts.sigma_min,
            sigma_max=opts.sigma_max,
            rho=opts.rho,
        )
    c.validation_kwargs = dnnlib.EasyDict(
        enabled=opts.val,
        every=opts.val_every or c.snapshot_ticks,
        num_images=opts.val_num,
        seed=opts.val_seed,
        batch=opts.val_batch,
        sampler=val_sampler_cfg,
        labels=opts.val_label,
        ref=opts.val_ref,
        ref_data=opts.val_ref_data,
        dump_images_dir=opts.val_dump_images_dir,
        overwrite=opts.val_overwrite,
        at_start=opts.val_at_start,
        teacher=opts.val_teacher,
    )

    # Train.
    # Remove non-API keys before invoking training loop.
    for _key in ['teacher']:
        if _key in c:
            del c[_key]
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

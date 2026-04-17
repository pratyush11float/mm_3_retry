import torch
from torch_utils import persistence
from torch_utils import training_stats

from functorch import jvp
from torch.nn.utils.stateless import functional_call

from .momentmatching_ops import sample_timesteps_mm, sample_conditional_posterior


@persistence.persistent_class
class EDMMomentMatchLoss:
    """Moment Matching Distillation loss (Algorithm 2: Alternating Optimization).

    Three models:
        g_eta  (student/generator) — ``net`` passed to __call__, trainable on odd steps.
        g_theta (teacher)          — ``self.teacher_net``, always frozen.
        g_phi   (auxiliary)        — ``self.aux_net``, trainable on even steps.

    Even steps update phi:
        L(phi) = w(s) * { ||x_tilde - g_phi(z_s)||^2 + ||g_theta(z_s) - g_phi(z_s)||^2 }
    Odd steps update eta:
        L(eta) = w(s) * x_tilde^T  sg[g_phi(z_s) - g_theta(z_s)]
    """

    def __init__(
        self,
        teacher_net,
        aux_net,
        k: int = 8,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.5,
        weight_mode: str = "edm",
        sync_dropout: bool = True,
        enable_stats: bool = True,
    ):
        assert k >= 1, "Student steps k must be >= 1"
        assert weight_mode in (
            "edm", "vlike", "flat",
            "snr", "snr+1", "karras", "sqrt_karras", "truncated-snr", "uniform",
        )

        self.teacher_net = teacher_net.eval().requires_grad_(False)
        self.aux_net = aux_net
        self.k = int(k)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.sigma_data = float(sigma_data)
        self.weight_mode = weight_mode
        self.sync_dropout = bool(sync_dropout)
        self.enable_stats = enable_stats

        self._step_n = 0

    # ------------------------------------------------------------------
    # Setters called by the training loop
    # ------------------------------------------------------------------
    def set_step_n(self, n: int) -> None:
        self._step_n = int(n)

    def set_run_dir(self, run_dir: str) -> None:
        self._run_dir = run_dir

    def set_global_kimg(self, kimg: float) -> None:
        self._global_kimg = float(kimg)

    # ------------------------------------------------------------------
    # Loss weighting  (evaluated at sigma_s per the paper)
    # ------------------------------------------------------------------
    def _weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """Per-sample weighting w(sigma). sigma shape [N] or [N,1,1,1]."""
        if self.weight_mode == "edm":
            return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        if self.weight_mode == "vlike":
            return (1.0 / (sigma ** 2)) + 1.0
        if self.weight_mode == "flat":
            return torch.ones_like(sigma)
        snr = 1.0 / (sigma ** 2 + 1e-20)
        if self.weight_mode == "snr":
            return snr
        if self.weight_mode == "snr+1":
            return snr + 1.0
        if self.weight_mode == "karras":
            return snr + (1.0 / (self.sigma_data ** 2))
        if self.weight_mode == "sqrt_karras":
            return torch.sqrt(sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data)
        if self.weight_mode == "truncated-snr":
            return torch.clamp(snr, min=1.0)
        assert self.weight_mode == "uniform"
        return torch.ones_like(sigma)

    # ------------------------------------------------------------------
    # Main loss
    # ------------------------------------------------------------------
    def __call__(self, net, images, labels=None, augment_pipe=None):
        device = images.device
        batch_size = images.shape[0]
        is_even = (self._step_n % 2 == 0)

        # Augmentation
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )

        # Step 1: sample timesteps
        ts = sample_timesteps_mm(
            batch_size=batch_size,
            k=self.k,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            device=device,
        )
        sigma_t_vec = ts["sigma_t"]  # [N] float64
        sigma_s_vec = ts["sigma_s"]  # [N] float64

        # Broadcast to [N,1,1,1] for BCHW ops
        sigma_t = sigma_t_vec.reshape(batch_size, 1, 1, 1)
        sigma_s = sigma_s_vec.reshape(batch_size, 1, 1, 1)

        # Step 2: forward diffusion  z_t = y + sigma_t * eps
        eps = torch.randn_like(y).to(torch.float64)
        y64 = y.to(torch.float64)
        z_t = y64 + sigma_t * eps  # float64
        z_t_f32 = z_t.float()

        # EDM weight evaluated at sigma_s
        weight = self._weight(sigma_s_vec.float()).reshape(batch_size, 1, 1, 1)

        if is_even:
            loss = self._loss_even(
                net, z_t_f32, sigma_t_vec, sigma_s_vec, sigma_t, sigma_s,
                labels, augment_labels, weight, batch_size,
            )
        else:
            loss = self._loss_odd(
                net, z_t_f32, sigma_t_vec, sigma_s_vec, sigma_t, sigma_s,
                labels, augment_labels, weight, batch_size,
            )

        # Stats
        if self.enable_stats:
            with torch.no_grad():
                training_stats.report('Loss/mm', loss)
                training_stats.report('MM/sigma_t', sigma_t_vec.float().mean())
                training_stats.report('MM/sigma_s', sigma_s_vec.float().mean())
                training_stats.report('MM/is_even', torch.as_tensor(float(is_even), device=device))

        return loss

    # ------------------------------------------------------------------
    # Even step: update auxiliary model phi
    # ------------------------------------------------------------------
    def _loss_even(
        self, net, z_t_f32, sigma_t_vec, sigma_s_vec, sigma_t, sigma_s,
        labels, augment_labels, weight, batch_size,
    ):
        # Step 3: student prediction under no_grad (student frozen on even steps)
        with torch.no_grad():
            x_tilde = net(
                z_t_f32, sigma_t_vec, labels, augment_labels=augment_labels,
            ).to(torch.float32)

        # Step 4: conditional posterior  z_s ~ q(z_s | z_t, x_tilde)
        z_s = sample_conditional_posterior(
            z_t_f32, x_tilde, sigma_t_vec, sigma_s_vec,
        )
        z_s_f32 = z_s.float()

        # Step 5: evaluate g_phi (with grad) and g_theta (no grad, synced dropout)
        if self.sync_dropout:
            rng_state = torch.cuda.get_rng_state()

        g_phi_zs = self.aux_net(
            z_s_f32, sigma_s_vec, labels, augment_labels=augment_labels,
        ).to(torch.float32)

        with torch.no_grad():
            if self.sync_dropout:
                torch.cuda.set_rng_state(rng_state)
            g_theta_zs = self.teacher_net(
                z_s_f32, sigma_s_vec, labels, augment_labels=augment_labels,
            ).to(torch.float32)

        # Step 6: L(phi) = w(s) * { ||x_tilde - g_phi||^2 + ||g_theta - g_phi||^2 }
        term1 = ((x_tilde.detach() - g_phi_zs) ** 2).sum(dim=[1, 2, 3])  # [N]
        term2 = ((g_theta_zs.detach() - g_phi_zs) ** 2).sum(dim=[1, 2, 3])  # [N]
        per_sample = weight.reshape(-1) * (term1 + term2)  # [N]
        return per_sample.reshape(batch_size, 1, 1, 1)

    # ------------------------------------------------------------------
    # Odd step: update student generator eta
    # ------------------------------------------------------------------
    def _loss_odd(
        self, net, z_t_f32, sigma_t_vec, sigma_s_vec, sigma_t, sigma_s,
        labels, augment_labels, weight, batch_size,
    ):
        # Step 3: student prediction WITH grad
        x_tilde = net(
            z_t_f32, sigma_t_vec, labels, augment_labels=augment_labels,
        ).to(torch.float32)

        # Step 4: conditional posterior  z_s ~ q(z_s | z_t, x_tilde)
        # Detach x_tilde for z_s computation (stop-gradient on z_s path)
        z_s = sample_conditional_posterior(
            z_t_f32, x_tilde.detach(), sigma_t_vec, sigma_s_vec,
        )
        z_s_f32 = z_s.float()

        # Step 5: evaluate g_phi and g_theta (both no grad, synced dropout)
        with torch.no_grad():
            if self.sync_dropout:
                rng_state = torch.cuda.get_rng_state()

            g_phi_zs = self.aux_net(
                z_s_f32, sigma_s_vec, labels, augment_labels=augment_labels,
            ).to(torch.float32)

            if self.sync_dropout:
                torch.cuda.set_rng_state(rng_state)

            g_theta_zs = self.teacher_net(
                z_s_f32, sigma_s_vec, labels, augment_labels=augment_labels,
            ).to(torch.float32)

        # Step 6: L(eta) = w(s) * x_tilde^T  sg[g_phi(z_s) - g_theta(z_s)]
        direction = (g_phi_zs - g_theta_zs).detach()
        dot = (x_tilde * direction).sum(dim=[1, 2, 3])  # [N]
        per_sample = weight.reshape(-1) * dot  # [N]
        return per_sample.reshape(batch_size, 1, 1, 1)

@persistence.persistent_class
class EDMInstantMomentMatchLoss:
    requires_two_batches = True

    def __init__(
        self,
        teacher_net,
        k=8,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        sigma_data=0.5,
        weight_mode='flat',
        precond_mode='identity',
        teacher_state_dump=None,
        sync_dropout=False,
    ):
        self.teacher_net = teacher_net.eval().to(next(teacher_net.parameters()).device)
        self.k = int(k)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.sigma_data = float(sigma_data)
        self.weight_mode = weight_mode
        self.precond_mode = precond_mode
        self.sync_dropout = sync_dropout

        self._teacher_param_names = [name for name, _ in self.teacher_net.named_parameters()]
        self._teacher_params = tuple(self.teacher_net.parameters())

        # Needed for teacher gradient / JVP calculations.
        for p in self._teacher_params:
            p.requires_grad_(True)

        self._adam_v = self._load_or_build_preconditioner(teacher_state_dump)
    
    def _weight(self, sigma):
        sigma = sigma.to(torch.float32)
        if self.weight_mode == 'edm':
            return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        if self.weight_mode == 'flat':
            return torch.ones_like(sigma)
        raise ValueError(f'Unknown weight_mode: {self.weight_mode}')
    
    def _load_or_build_preconditioner(self, teacher_state_dump):
        if self.precond_mode == 'identity':
            return None
        if teacher_state_dump is None:
            return None

        try:
            data = torch.load(teacher_state_dump, map_location='cpu')
        except Exception:
            return None

        opt_state = data.get('optimizer_state', None)
        if opt_state is None:
            return None

        state_values = list(opt_state.get('state', {}).values())
        if len(state_values) != len(self._teacher_params):
            return None

        out = {}
        for name, st in zip(self._teacher_param_names, state_values):
            v = st.get('exp_avg_sq', None)
            if v is None:
                return None
            out[name] = v.detach().clone().float()
        return out
    
    def _precondition(self, grads):
        out = []
        for name, p, g in zip(self._teacher_param_names, self._teacher_params, grads):
            if g is None:
                g = torch.zeros_like(p)

            if self.precond_mode == 'identity' or self._adam_v is None:
                out.append(-g.detach())
            else:
                v = self._adam_v[name].to(device=g.device, dtype=g.dtype)
                out.append(-(g.detach() / (v.sqrt() + 1e-8)))
        return tuple(out)
    
    def _teacher_grad_on_generated_batch(self, net, images, labels=None, augment_pipe=None):
        device = images.device
        batch_size = images.shape[0]

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        ts = sample_timesteps_mm(
            batch_size=batch_size,
            k=self.k,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            device=device,
        )
        sigma_t_vec = ts['sigma_t']
        sigma_s_vec = ts['sigma_s']

        sigma_t = sigma_t_vec.reshape(batch_size, 1, 1, 1)
        eps = torch.randn_like(y, dtype=torch.float64)
        z_t = y.to(torch.float64) + sigma_t * eps
        z_t_f32 = z_t.float()

        with torch.no_grad():
            x_tilde = net(
                z_t_f32, sigma_t_vec, labels, augment_labels=augment_labels,
            ).to(torch.float32)

        z_s = sample_conditional_posterior(
            z_t_f32, x_tilde, sigma_t_vec, sigma_s_vec,
        ).float()

        weight = self._weight(sigma_s_vec).reshape(batch_size, 1, 1, 1)

        teacher_pred = self.teacher_net(
            z_s, sigma_s_vec, labels, augment_labels=augment_labels,
        ).to(torch.float32)

        teacher_loss = (weight * ((teacher_pred - x_tilde) ** 2)).sum() / batch_size

        grads = torch.autograd.grad(
            teacher_loss,
            self._teacher_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        return tuple(
            torch.zeros_like(p) if g is None else g
            for p, g in zip(self._teacher_params, grads)
        )
    def _teacher_jvp_direction(self, z_s, sigma_s_vec, labels, augment_labels, nu):
        params = {name: p for name, p in self.teacher_net.named_parameters()}
        buffers = {name: b for name, b in self.teacher_net.named_buffers()}
        state = {}
        state.update(params)
        state.update(buffers)

        tangent_state = {}
        for name, v in zip(self._teacher_param_names, nu):
            tangent_state[name] = v
        for name, b in self.teacher_net.named_buffers():
            tangent_state[name] = torch.zeros_like(b)
        
        def f(state_dict):
            out = functional_call(
                self.teacher_net,
                state_dict,
                (z_s, sigma_s_vec, labels),
                {'augment_labels': augment_labels},
            )
            return out.to(torch.float32)

        _, jvp_out = jvp(f, (state,), (tangent_state,))
        return jvp_out.detach()
    
    def __call__(self, net, images_a, labels_a=None, images_b=None, labels_b=None, augment_pipe=None):
        if images_b is None:
            raise ValueError('Algorithm 3 requires two independent minibatches.')

        # ---------- Batch A: teacher gradient ----------
        grads_a = self._teacher_grad_on_generated_batch(
            net=net, images=images_a, labels=labels_a, augment_pipe=augment_pipe
        )
        nu = self._precondition(grads_a)

        # ---------- Batch B: student loss ----------
        device = images_b.device
        batch_size = images_b.shape[0]

        y_b, augment_labels_b = augment_pipe(images_b) if augment_pipe is not None else (images_b, None)

        ts_b = sample_timesteps_mm(
            batch_size=batch_size,
            k=self.k,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            device=device,
        )
        sigma_t_b = ts_b['sigma_t']
        sigma_s_b = ts_b['sigma_s']

        sigma_t_b_4d = sigma_t_b.reshape(batch_size, 1, 1, 1)
        eps_b = torch.randn_like(y_b, dtype=torch.float64)
        z_t_b = y_b.to(torch.float64) + sigma_t_b_4d * eps_b
        z_t_b_f32 = z_t_b.float()

        x_tilde_b = net(
            z_t_b_f32, sigma_t_b, labels_b, augment_labels=augment_labels_b,
        ).to(torch.float32)

        z_s_b = sample_conditional_posterior(
            z_t_b_f32, x_tilde_b.detach(), sigma_t_b, sigma_s_b,
        ).float()

        direction_b = self._teacher_jvp_direction(
            z_s=z_s_b,
            sigma_s_vec=sigma_s_b,
            labels=labels_b,
            augment_labels=augment_labels_b,
            nu=nu,
        )

        weight_b = self._weight(sigma_s_b).reshape(batch_size, 1, 1, 1)
        per_sample = weight_b.reshape(-1) * (x_tilde_b * direction_b).sum(dim=[1, 2, 3])

        training_stats.report('Loss/mm', per_sample.mean())
        training_stats.report('MM/sigma_t', sigma_t_b.float().mean())
        training_stats.report('MM/sigma_s', sigma_s_b.float().mean())

        return per_sample.reshape(batch_size, 1, 1, 1)
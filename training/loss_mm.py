import torch
from torch_utils import persistence
from torch_utils import training_stats

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

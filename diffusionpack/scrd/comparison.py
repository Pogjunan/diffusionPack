import torch
from torch import nn
from tqdm.auto import tqdm

from ..backbone.ddpm_unet import DDPMUNet
from ..backbone.prediction_unet import PredictionUNet
from ..diffusion import check_shape, step_value


class ConditionalDDPM(DDPMUNet):

    def __init__(self, history_steps, future_steps, map_channels, **settings):
        super().__init__(in_channels=(history_steps + future_steps) * map_channels, **settings)
        self.target_channels = future_steps * map_channels
        self.condition_channels = history_steps * map_channels
        self.conv_out = nn.Conv2d(self.conv_out.in_channels, self.target_channels, 3, padding=1)

    def forward(self, noisy, t, condition):
        if noisy.shape[1] != self.target_channels or condition.shape[1] != self.condition_channels:
            raise ValueError("History or future channel count differs from the model")
        return super().forward(torch.cat([noisy, condition], dim=1), t)


def make_model(config):
    frames = dict(history_steps=config.history_steps, future_steps=config.future_steps,
                  map_channels=config.map_channels)
    if config.backbone == "scrd":
        from .unet import SCRDUNet
        if config.noise_conditioning != "timestep":
            raise ValueError("SCRD uses discrete timesteps; gamma is a separate DDPM comparison")
        return SCRDUNet(**frames, widths=config.widths, window=config.window,
                        tcu_depth=config.tcu_depth, time_dim=config.time_dim,
                        norm_groups=config.norm_groups, output_norm_groups=config.output_norm_groups,
                        use_tcu=config.use_tcu)
    if config.backbone != "ddpm":
        raise ValueError("backbone must be ddpm or scrd")
    if config.time_input != "index":
        raise ValueError("The existing DDPM backbone expects index time or gamma")
    settings = dict(resolution=config.resolution, base_channels=config.base_channels,
                    channel_mult=config.channel_mult, num_res_blocks=config.num_res_blocks,
                    attention_resolutions=config.attention_resolutions, dropout=config.dropout)
    model_class = PredictionUNet if config.noise_conditioning == "gamma" else ConditionalDDPM
    return model_class(**frames, **settings)


class ForecastDiffusion(nn.Module):

    def __init__(self, model, *, future_steps=4, map_channels=1, steps=1000,
                 beta_start=0.0001, beta_end=0.02, target="residual",
                 noise_conditioning="timestep", time_input="index", reverse_variance="posterior"):
        super().__init__()
        if type(steps) is not int or steps < 1 or not 0 < beta_start <= beta_end < 1:
            raise ValueError("Use a positive number of steps and 0 < beta_start <= beta_end < 1")
        if min(future_steps, map_channels) < 1:
            raise ValueError("Future steps and map channels must be positive")
        if target not in ("future", "residual") or noise_conditioning not in ("timestep", "gamma"):
            raise ValueError("Choose future/residual and timestep/gamma")
        if time_input not in ("index", "fraction") or reverse_variance not in ("beta", "posterior"):
            raise ValueError("Choose index/fraction time and beta/posterior variance")
        self.model = model
        self.future_steps, self.map_channels, self.steps = future_steps, map_channels, steps
        self.target, self.noise_conditioning = target, noise_conditioning
        self.time_input, self.reverse_variance = time_input, reverse_variance
        betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
        alphas = 1 - betas
        alpha_bars = alphas.cumprod(0)
        previous = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bars[:-1]])
        for name, value in dict(betas=betas, alphas=alphas, alpha_bars=alpha_bars,
                                alpha_bars_prev=previous,
                                posterior_var=betas * (1 - previous) / (1 - alpha_bars)).items():
            self.register_buffer(name, value)

    def last_frame(self, condition):
        return condition[:, -self.map_channels:].repeat(1, self.future_steps, 1, 1)

    def make_target(self, future, condition):
        last = self.last_frame(condition)
        check_shape(future, last)
        return future - last if self.target == "residual" else future

    def q_sample(self, clean, t, noise):
        check_shape(clean, noise)
        return (step_value(self.alpha_bars.sqrt(), t, clean) * clean
                + step_value((1 - self.alpha_bars).sqrt(), t, clean) * noise)

    def model_time(self, t, gamma=None):
        if self.noise_conditioning == "gamma":
            return self.alpha_bars[t].float() if gamma is None else gamma.reshape(-1)
        return (t.float() + 1) / self.steps if self.time_input == "fraction" else t

    def predict_noise(self, noisy, t, condition, features=None, gamma=None):
        time = self.model_time(t, gamma)
        if features is None:
            predicted = self.model(noisy, time, condition)
        else:
            predicted = self.model(noisy, time, condition, features)
        check_shape(predicted, noisy)
        return predicted

    def loss(self, future, condition, t=None, noise=None):
        clean = self.make_target(future, condition)
        if t is None:
            t = torch.randint(self.steps, (len(clean),), device=clean.device)
        if noise is None:
            noise = torch.randn_like(clean)
        check_shape(clean, noise)
        gamma = None
        if self.noise_conditioning == "gamma":
            low = step_value(self.alpha_bars, t, clean)
            high = step_value(self.alpha_bars_prev, t, clean)
            gamma = torch.lerp(low, high, torch.rand_like(low))
            noisy = gamma.sqrt() * clean + (1 - gamma).sqrt() * noise
        else:
            noisy = self.q_sample(clean, t, noise)
        predicted = self.predict_noise(noisy, t, condition, gamma=gamma)
        return (predicted.float() - noise.float()).square().mean()

    def predict_x0(self, noisy, t, predicted):
        check_shape(noisy, predicted)
        return (noisy - step_value((1 - self.alpha_bars).sqrt(), t, noisy) * predicted) / step_value(
            self.alpha_bars.sqrt(), t, noisy)

    def p_step(self, noisy, t, predicted, noise=None):
        check_shape(noisy, predicted)
        weight = step_value(self.betas / (1 - self.alpha_bars).sqrt(), t, noisy)
        mean = (noisy - weight * predicted) / step_value(self.alphas.sqrt(), t, noisy)
        last = (t == 0).reshape(-1, 1, 1, 1)
        mean = torch.where(last, self.predict_x0(noisy, t, predicted), mean)
        variance = self.betas if self.reverse_variance == "beta" else self.posterior_var
        if noise is None:
            noise = torch.randn_like(noisy)
        check_shape(noisy, noise)
        if noise.device != noisy.device or noise.dtype != noisy.dtype:
            raise ValueError("Noise must match the state device and dtype")
        return mean + (~last) * step_value(variance.sqrt(), t, noisy) * noise

    @torch.no_grad()
    def sample(self, condition, *, progress=False, initial_noise=None, generator=None):
        was_training = self.model.training
        self.model.eval()
        try:
            shape = (len(condition), self.future_steps * self.map_channels, *condition.shape[-2:])
            if initial_noise is not None:
                if initial_noise.shape != shape or initial_noise.device != condition.device or initial_noise.dtype != condition.dtype:
                    raise ValueError("Initial noise must match forecast shape, device and dtype")
                state = initial_noise.clone()
            else:
                state = torch.randn(shape, device=condition.device, dtype=condition.dtype, generator=generator)
            features = self.model.prepare_condition(condition) if hasattr(self.model, "prepare_condition") else None
            for index in tqdm(range(self.steps - 1, -1, -1), disable=not progress, desc="DDPM", leave=False):
                t = torch.full((len(condition),), index, device=condition.device, dtype=torch.long)
                predicted = self.predict_noise(state, t, condition, features)
                noise = torch.zeros_like(state) if index == 0 else torch.randn(
                    shape, device=state.device, dtype=state.dtype, generator=generator)
                state = self.p_step(state, t, predicted, noise)
            if self.target == "residual":
                state = state + self.last_frame(condition)
            if not torch.isfinite(state).all():
                raise FloatingPointError("Non-finite forecast; do not clip or replace it")
            return state
        finally:
            self.model.train(was_training)


def make_diffusion(config):
    return ForecastDiffusion(make_model(config), future_steps=config.future_steps,
                             map_channels=config.map_channels, steps=config.diffusion_steps,
                             beta_start=config.beta_start, beta_end=config.beta_end, target=config.target,
                             noise_conditioning=config.noise_conditioning, time_input=config.time_input,
                             reverse_variance=config.reverse_variance)

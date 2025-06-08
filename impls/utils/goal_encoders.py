import flax.linen as nn
import jax.numpy as jnp
import distrax

class VIBEncoder(nn.Module):
    encoder: nn.Module
    beta: int = 1.0
    rep_dim: int = 256
    log_std_min: int = -5

    @nn.compact
    def __call__(self, goal, rng):
        goal = self.encoder(goal)
        mean, log_stds = nn.Dense(self.rep_dim)(goal), nn.Dense(self.rep_dim)(goal)
        log_stds = jnp.clip(log_stds, min=self.log_std_min)
        stds = jnp.exp(log_stds)

        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_stds))
        prior = distrax.MultivariateNormalDiag(loc=jnp.zeros_like(mean))
        z = dist.sample(seed=rng)
        # z, log_probs = dist.sample_and_log_prob(seed=rng)
        # kl_loss = log_probs - prior.log_prob(z)

        # return z, (self.beta * kl_loss).mean()

        mean_term = (mean * mean).sum(axis=-1)
        sd_trace = jnp.sum(stds, axis=-1)
        log_det_stds = jnp.log(jnp.prod(stds, axis=-1))
        kl = 0.5 * (mean_term + sd_trace - mean.shape[-1] - log_det_stds)

        return z, (self.beta * kl).mean()

goal_encoders = {
    'vib': VIBEncoder,
}
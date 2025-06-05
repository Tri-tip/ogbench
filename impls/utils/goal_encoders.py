import flax.linen as nn
import jax.numpy as jnp
import distrax

class VIBEncoder(nn.Module):
    encoder: nn.Module
    beta: int = 1.0
    rep_dim: int = 256

    @nn.compact
    def __call__(self, goal, rng):
        goal = self.encoder(goal)
        mean, sd = nn.Dense(self.rep_dim)(goal), nn.Dense(self.rep_dim)(goal)
        sd = nn.softplus(sd - 5.0)

        # # reparameterization trick
        # eps = jax.random.normal(new_rng, sd.shape)
        # z = eps * sd + mean
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=sd)
        prior = distrax.MultivariateNormalDiag(loc=jnp.zeros_like(mean))
        z, log_probs = dist.sample_and_log_prob(seed=rng)
        kl_loss = log_probs - prior.log_prob(z)

        return z, (self.beta * kl_loss).mean()


goal_encoders = {
    'vib': VIBEncoder,
}
from typing import Any, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron (MLP).

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x

class Param(nn.Module):
    """Scalar parameter module."""

    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param('value', init_fn=lambda key: jnp.full((), self.init_value))


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class RunningMeanStd(flax.struct.PyTreeNode):
    """Running mean and standard deviation.

    Attributes:
        eps: Epsilon value to avoid division by zero.
        mean: Running mean.
        std: Running standard deviation.
        alpha: Smoothing factor for the running mean and variance.
    """

    eps: Any = 1e-6
    mean: Any = 0.0
    std: Any = 1.0
    alpha: Any = 0.01

    def normalize(self, batch, normalize_mean=True):
        if normalize_mean:
            batch = (batch - self.mean) / (self.std + self.eps)
        else:
            batch = batch / (self.std + self.eps)

        return batch

    def update(self, mean, std):
        new_mean = self.alpha * mean + (1 - self.alpha) * self.mean
        new_std = self.alpha * std + (1 - self.alpha) * self.std

        return self.replace(mean=new_mean, std=new_std)

class GCDeterActor(nn.Module):
    """Goal-conditioned deterministic actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        mlp_class: MLP class.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action distribution with tanh.
        mean_tanh_squash: Whether to squash the mean net with tanh (the action distribution can still be unbounded).
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    mlp_class: Any = MLP
    layer_norm: bool = False
    tanh_squash: bool = False
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = self.mlp_class(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Scaling factor for the standard deviation.
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)
        means = self.mean_net(outputs)
        if self.tanh_squash:
            means = jnp.tanh(means)
        return means

class GCActor(nn.Module):
    """Goal-conditioned actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        mlp_class: MLP class.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action distribution with tanh.
        mean_tanh_squash: Whether to squash the mean net with tanh (the action distribution can still be unbounded).
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    mlp_class: Any = MLP
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    mean_tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = self.mlp_class(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Scaling factor for the standard deviation.
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.mean_tanh_squash:
            means = jnp.tanh(means)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class GCDiscreteActor(nn.Module):
    """Goal-conditioned actor for discrete actions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        mlp_class: MLP class.
        layer_norm: Whether to apply layer normalization.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    mlp_class: Any = MLP
    layer_norm: bool = False
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = self.mlp_class(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.logit_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Inverse scaling factor for the logits (set to 0 to get the argmax).
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        logits = self.logit_net(outputs)

        distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))

        return distribution


class GCValue(nn.Module):
    """Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        output_dim: Output dimension (set to None for scalar output).
        mlp_class: MLP class.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    output_dim: int = None
    mlp_class: Any = MLP
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = self.mlp_class
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        output_dim = self.output_dim if self.output_dim is not None else 1
        value_net = mlp_class((*self.hidden_dims, output_dim), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, goals=None, actions=None):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
        """

        if self.encoder is not None:
            if goals is None:
                inputs = [self.encoder(observations)]
            else:
                inputs = [self.encoder(observations), goals]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs)
        if self.output_dim is None:
            v = v.squeeze(-1)

        return v

class ActorVectorField(nn.Module):
    """Actor vector field for flow policies.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        mlp_class: MLP class.
        activate_final: Whether to apply activation to the final layer.
        layer_norm: Whether to apply layer normalization.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    mlp_class: Any = MLP
    activate_final: bool = False
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = self.mlp_class(
            (*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm
        )

    @nn.compact
    def __call__(self, observations, goals=None, actions=None, times=None, is_encoded=False):
        """Return the current vector.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Current actions.
            times: Current times (optional).
            is_encoded: Whether the inputs are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            if goals is None:
                inputs = self.encoder(observations)
            else:
                inputs = jnp.concatenate([self.encoder(observations), goals], axis=-1)
        else:
            if goals is None:
                inputs = observations
            else:
                inputs = jnp.concatenate([observations, goals], axis=-1)
        if times is None:
            inputs = jnp.concatenate([inputs, actions], axis=-1)
        else:
            inputs = jnp.concatenate([inputs, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v

class ConvBlock(nn.Module):
    ch: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.ch, (3,3), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        residual = x
        
        x = nn.Conv(self.ch, (3,3), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(self.ch, (3,3), padding='SAME')(x)
        x = nn.LayerNorm()(x)

        return residual + x

class Dynamics(nn.Module):
    """Dynamics model.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        output_dim: Output dimension (set to None for scalar output).
        mlp_class: MLP class.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
    """

    hidden_dims: Sequence[int]
    output_dim: int = None
    mlp_class: Any = MLP
    layer_norm: bool = True
    num_ensembles: int = 1
    delta_pred: bool = True
    stochastic: bool = False

    def setup(self):
        mlp_class = self.mlp_class
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        output_dim = self.output_dim if self.output_dim is not None else 1
        if self.stochastic:
            dynamics_net = mlp_class((*self.hidden_dims, 2 * output_dim), activate_final=False, layer_norm=self.layer_norm)
        else:
            dynamics_net = mlp_class((*self.hidden_dims, output_dim), activate_final=False, layer_norm=self.layer_norm)

        self.dynamics_net = dynamics_net

    def __call__(self, observations, actions):
        """Return the predicted next states.

        Args:
            observations: Observations.
            actions: Actions.
        """
        inputs = []
        inputs.append(observations)
        inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        pred = self.dynamics_net(inputs)
        if self.stochastic:
            means, log_stds = pred[..., :self.output_dim], pred[..., self.output_dim:]
            min_logstd, max_logstd = -5.0, 1.0
            log_stds = jax.nn.sigmoid(log_stds) * (max_logstd - min_logstd) + min_logstd
            if self.delta_pred: means = observations + means
            distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
            return distribution

        if self.delta_pred:
            pred = observations + pred

        return pred
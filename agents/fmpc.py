import copy
from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCValue, RunningMeanStd, DynamicsVectorField, ActorVectorField


class FMPCAgent(flax.struct.PyTreeNode):
    """Flow version of DMPC (Zhou et al., 2025) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def bce_loss(pred_logit, target):
        """Compute the BCE loss."""
        log_pred = jax.nn.log_sigmoid(pred_logit)
        log_not_pred = jax.nn.log_sigmoid(-pred_logit)
        loss = -(log_pred * target + log_not_pred * (1 - target))
        return loss

    def dynamics_loss(self, batch, grad_params, rng):
        """Compute the dynamics loss."""
        batch_size, ob_dim = batch['next_observations'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, ob_dim))
        x_1 = batch['next_observations']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('dynamics_flow')(
            batch['observations'], batch['actions'], x_t, t, params=grad_params
        )
        flow_loss = jnp.mean((pred - vel) ** 2)

        return flow_loss, {
            'flow_loss': flow_loss,
            'vel': jnp.linalg.norm(vel, axis=-1).mean(),
        }
   
    def bc_actor_loss(self, batch, grad_params, rng):
        """Compute the BC flow actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(
            batch['observations'], goals=batch['actor_goals'], actions=x_t, times=t, params=grad_params
        )
        flow_loss = jnp.mean((pred - vel) ** 2)

        return flow_loss, {
            'flow_loss': flow_loss,
            'vel': jnp.linalg.norm(vel, axis=-1).mean(),
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss."""
        batch_size, ob_dim = batch['observations'].shape

        q = self.network.select('critic')(batch['next_observations'], batch['value_goals'], batch['actions'], params=grad_params)

        discount = self.config['discount'] ** self.config['action_chunking'] 
        reward = batch['rewards']
        mask = batch['masks']
        next_v = self.network.select('value')(batch['next_observations'][:, -ob_dim:], batch['value_goals'])
        target = reward + discount * mask * next_v

        if self.config['value_loss'] == 'bce':
            q_logit = q
            q = jax.nn.sigmoid(q_logit)

        if self.config['value_loss'] == 'squared':
            critic_loss = ((q - target) ** 2).mean()
        elif self.config['value_loss'] == 'bce':
            # Clip targets to [0, 1] for BCE loss.
            target = jnp.clip(target, 0, 1)
            critic_loss = (self.bce_loss(q_logit, target)).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }
 

    def value_loss(self, batch, grad_params, rng):
        """Compute the value loss (including the critic loss)."""
        v = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        if self.config['value_loss'] == 'bce':
            v_logit = v
            v = jax.nn.sigmoid(v_logit)

        if self.config['value_loss'] == 'squared':
            value_loss = ((v - batch['returns_to_go']) ** 2).mean()
        elif self.config['value_loss'] == 'bce':
            # Clip targets to [0, 1] for BCE loss.
            target = jnp.clip(batch['returns_to_go'], 0, 1)
            value_loss = (self.bce_loss(v_logit, target)).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, dynamics_rng, bc_actor_rng, rollout_rng, value_rng, critic_rng = jax.random.split(rng, 6)

        dynamics_loss, dynamics_info = self.dynamics_loss(batch, grad_params, dynamics_rng)
        for k, v in dynamics_info.items():
            info[f'dynamics/{k}'] = v

        bc_actor_loss, bc_actor_info = self.bc_actor_loss(batch, grad_params, bc_actor_rng)
        for k, v in bc_actor_info.items():
            info[f'bc_actor/{k}'] = v

        value_loss, value_info = self.value_loss(batch, grad_params, value_rng)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        loss = dynamics_loss + bc_actor_loss + value_loss + critic_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def compute_metrics(self, batch, rng=None):
        action_rng, dynamics_rng = jax.random.split(rng)

        actions, _ = self.sample_actions(batch['observations'], batch['actor_goals'], seed=action_rng, mode='train')
        action_mse = jnp.mean((actions - batch['actions']) ** 2)

        noises = jax.random.normal(dynamics_rng, batch['next_observations'].shape)
        next_observations = self.sample_dynamics(batch['observations'], batch['actions'], noises)
        dynamics_mse = jnp.mean((next_observations - batch['next_observations']) ** 2)

        info = {
            'action_mse': action_mse,
            'dynamics_mse': dynamics_mse,
        }

        return info

    @partial(jax.jit, static_argnames=['mode'])
    def sample_actions(
            self,
            observations,
            goals=None,
            seed=None,
            temperature=1.0,
            mode='test',
    ):
        """Sample actions from the actor."""
        num_samples = self.config['num_samples']
        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), num_samples, axis=0)
        n_goals = jnp.repeat(jnp.expand_dims(goals, 0), num_samples, axis=0) if goals is not None else None

        n_noises = jax.random.normal(seed, (num_samples, *observations.shape[:-1], self.config['action_dim']))
        n_actions = self.sample_flow_actions(n_observations, n_goals, n_noises)

        n_next_noises = jax.random.normal(seed, (num_samples, *observations.shape[:-1], self.config['next_ob_dim']))
        n_next_observations = self.sample_dynamics(n_observations, n_actions, n_next_noises)

        q = self.network.select('critic')(n_next_observations, n_goals, n_actions)
        if len(observations.shape) == 2:
            actions = n_actions[jnp.argmax(q, axis=0), jnp.arange(observations.shape[0])]
        else:
            actions = n_actions[jnp.argmax(q)]
        actions = jnp.clip(actions, -1, 1)

        if mode == 'test':
            ac_actor = self.config['action_chunking_actor'] if 'action_chunking_actor' in self.config and self.config['action_chunking_actor'] > 0 else self.config['action_chunking']
            action_dim = actions.shape[-1] // ac_actor
            actions = actions[..., :action_dim * self.config['action_chunking_test']]

        return actions, {}

    @jax.jit
    def sample_flow_actions(
            self,
            observations,
            goals,
            noises,
    ):
        """Sample actions from the BC flow policy."""
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_flow')(observations, goals=goals, actions=actions, times=t)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions
    
    @jax.jit
    def sample_dynamics(
            self,
            observations,
            actions,
            noises,
    ):
        """Sample actions from the BC flow policy."""
        next_ob = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('dynamics_flow')(observations, actions, next_ob, t)
            next_ob = next_ob + vels / self.config['flow_steps']
        return next_ob

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_next_observations = example_batch['next_observations']
        ex_goals = example_batch['actor_goals']
        ex_times = ex_actions[..., :1]
        ob_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]
        next_ob_dim = ex_next_observations.shape[-1]

        # Define networks.
        dynamics_flow_def = DynamicsVectorField(
            hidden_dims=config['dynamics_hidden_dims'],
            output_dim=next_ob_dim,
            layer_norm=config['layer_norm'],
        )
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
        )
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
        )

        network_info = dict(
            dynamics_flow=(dynamics_flow_def, (ex_observations, ex_actions, ex_next_observations, ex_times)),
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_next_observations, ex_goals, ex_actions)),
            actor_flow=(actor_flow_def, (ex_observations, ex_goals, ex_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params

        config['ob_dim'] = ob_dim
        config['action_dim'] = action_dim
        config['next_ob_dim'] = next_ob_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='fmpc',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor network hidden dimensions.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Value network hidden dimensions.
            dynamics_hidden_dims=(1024, 1024, 1024, 1024),  # Dynamics network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.999,  # Discount factor.
            tau=0.005,  # Target network update rate.
            ob_dim=ml_collections.config_dict.placeholder(int),  # Observation dimension (set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (set automatically).
            next_ob_dim=ml_collections.config_dict.placeholder(int),  # Observation dimension (set automatically).
            rollout_batch_size=1024,  # Batch size for model-based rollouts.
            lam=1.0,  # GAE lambda.
            target_network=True,  # Whether to use target networks.
            value_loss='bce',  # Value loss type ('squared' or 'bce').
            flow_steps=10,  # Number of flow steps.
            num_samples=64,  # Number of samples for the actor.
            # Dataset hyperparameters.
            dataset_class='ACGCDataset',  # Dataset class name.
            action_chunking=32,  # Subgoal steps.
            chunk_next_obs=True,
            return_to_go=True,
            action_chunking_actor=1, # Action chunking
            action_chunking_test=1, 

            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=False,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.5,  # Probability of using a random state as the actor goal.
            actor_geom_sample=True,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
        )
    )
    return config

import copy
from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Dynamics, GCValue, ActorVectorField

sg = jax.lax.stop_gradient

class ACModelAgent(flax.struct.PyTreeNode):
    """Model-based flow rejection sampling with Action Chunking (MBFRS + AC) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def recon_loss(self, pred, target):
        return jnp.square(pred - target).mean()

    def dynamics_loss(self, batch, grad_params, rng):
        """Compute the dynamics loss."""
        pred_next_ob = self.network.select('dynamics')(batch['observations'], batch['actions'], params=grad_params)
        dynamics_loss = jnp.square(pred_next_ob - batch['next_observations']).mean()

        return dynamics_loss, {
            'dynamics_loss': dynamics_loss,
        }

    def success_loss(self, batch, grad_params, rng):
        """Compute the success prediction loss."""
        succ_logit = self.network.select('success')(batch['observations'], batch['value_goals'], batch['actions'], params=grad_params)

        success = jax.nn.sigmoid(succ_logit)
        success_target = 1 - batch['masks']
        success_loss = self.bce_loss(succ_logit, success_target).mean()

        return success_loss, {
            'success_loss': success_loss,
            'success_mean': success.mean(),
            'success_max': success.max(),
            'success_min': success.min(),
            'accuracy_0.5': jnp.mean((success > 0.5) == (success_target > 0.5)),
            'accuracy_0.7': jnp.mean((success > 0.7) == (success_target > 0.5)),
            'accuracy_0.9': jnp.mean((success > 0.9) == (success_target > 0.5)),
            'pos_acc': jnp.sum(jnp.logical_and(success > 0.5, success_target > 0.5).astype(jnp.float32)) / jnp.sum((success_target > 0.5).astype(jnp.float32)),
            'neg_acc': jnp.sum(jnp.logical_and(success < 0.5, success_target < 0.5).astype(jnp.float32)) / jnp.sum((success_target < 0.5).astype(jnp.float32)),
            'data_success_mean': (1 - batch['masks']).mean(),
        }

    def reward_loss(self, batch, grad_params, rng):
        """Compute the reward prediction loss."""
        rewards = self.network.select('reward')(batch['observations'], batch['value_goals'], batch['actions'], params=grad_params)

        rewards_target = batch['rewards']
        reward_loss = jnp.square(rewards_target - rewards).mean()

        return reward_loss, {
            'reward_loss': reward_loss,
            'reward_mean': rewards.mean(),
            'reward_max': rewards.max(),
            'reward_min': rewards.min(),
            'data_reward_mean': batch['rewards'].mean(),
        }

    def rollout(self, batch, grad_params, rng):
        """Perform model-based rollouts."""
        init_ob = batch['observations'][: self.config['rollout_batch_size']]
        if batch['value_goals'] is None: 
            goal = None
            batch['rollout_goals'] = None
        else: 
            goal = batch['value_goals'][: self.config['rollout_batch_size']]
            batch['rollout_goals'] = jnp.repeat(goal[None], self.config['rollout_steps'], axis=0) 

        def rollout_step(carry, _):
            ob, rng = carry
            rng, action_rng = jax.random.split(rng, 2)

            action, _ = self.sample_actions(ob, goal, action_rng, mode='train')
            next_ob = self.network.select('dynamics')(ob, action)

            return (next_ob, rng), {
                'observations': ob,
                'actions': action,
                'next_observations': next_ob,
            }

        rng, rollout_rng = jax.random.split(rng)
        init_carry = (init_ob, rollout_rng)
        final_carry, rollout_data = jax.lax.scan(rollout_step, init_carry, None, length=self.config['rollout_steps'])

        batch['rollout_observations'] = rollout_data['observations']
        batch['rollout_next_observations'] = rollout_data['next_observations']
        batch['rollout_actions'] = rollout_data['actions']
        if goal is None:
            success = jax.nn.sigmoid(self.network.select('success')(batch['rollout_observations'], batch['rollout_goals'], batch['rollout_actions']))
            success = jnp.where(success > 0.5, jnp.ones_like(success), jnp.zeros_like(success))
            rewards = self.network.select('reward')(batch['rollout_observations'], batch['rollout_goals'], batch['rollout_actions'])

            batch['rollout_rewards'] = rewards 
            batch['rollout_masks'] = 1.0 - success
        else:
            if self.config['learn_success']:
                success = jax.nn.sigmoid(self.network.select('success')(batch['rollout_observations'], batch['rollout_goals'], batch['rollout_actions']))
                success = jnp.where(success > 0.5, jnp.ones_like(success), jnp.zeros_like(success))
            else:
                success = self.success_fn(batch['rollout_observations'], goal)

            batch['rollout_rewards'] = success - (1.0 if self.config['gc_negative'] else 0.0)
            batch['rollout_masks'] = 1.0 - success
   
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        losses, info = {}, {}
        rng = rng if rng is not None else self.rng

        rng, dynamics_rng, bc_actor_rng, rollout_rng, value_rng, critic_rng, success_rng = jax.random.split(rng, 7)

        dynamics_loss, dynamics_info = self.dynamics_loss(batch, grad_params, dynamics_rng)
        losses['dynamics'] = dynamics_loss
        for k, v in dynamics_info.items():
            info[f'dynamics/{k}'] = v

        if self.config['learn_success']: 
            success_loss, success_info = self.success_loss(batch, grad_params, success_rng)
            for k, v in success_info.items():
                info[f'success/{k}'] = v
            losses['success'] = success_loss

            if 'rewards' in batch:
                reward_loss, reward_info = self.reward_loss(batch, grad_params, success_rng)
                for k, v in reward_info.items():
                    info[f'reward/{k}'] = v
                losses['reward'] = reward_loss

        loss = sum(losses.values())
        return loss, info

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
        actions, _ = self.sample_actions(batch['observations'], batch['actor_goals'], seed=rng, mode='train')
        mse = jnp.mean((actions - batch['actions']) ** 2)

        info = {
            'mse': mse,
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
        seed, encode_seed = jax.random.split(seed, 2)
        actions = jax.random.uniform(seed, (*observations.shape[:-1], self.config['action_dim']))
        if mode == 'test':
            action_dim = actions.shape[-1] // self.config['action_chunking']
            actions = actions[..., :action_dim * self.config['action_chunking_test']]
        return actions, {}

    @jax.jit
    def rollout_metrics(self, batch, rng):
        # batch = {
        #   'observations': [T,B,*d_ob]
        #   'actions': [T,B,d_ac]
        # }

        def rollout_step(carry, action):
            ob, rng = carry
            rng, action_rng = jax.random.split(rng, 2)

            next_ob = self.network.select('dynamics')(ob, action)

            return (next_ob, rng), {
                'observations': ob,
            }

        rng, rollout_rng = jax.random.split(rng)
        init_carry = (batch['observations'][0], rollout_rng)
        final_carry, rollout_data = jax.lax.scan(rollout_step, init_carry, batch['actions'])

        mse = jnp.square(batch['observations'] - rollout_data['observations']).mean(axis=(1, 2))
        return {
            'mse': mse
        }


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
        ex_goals = example_batch['actor_goals']
        ex_times = ex_actions[..., :1]
        ob_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]

        # Define networks.
        dynamics_def = Dynamics(
            hidden_dims=config['dynamics_hidden_dims'],
            output_dim=ob_dim,
            layer_norm=config['layer_norm'],
        )
        network_info = dict(
            dynamics=(dynamics_def, (ex_observations, ex_actions)),
       )

        if config['learn_success']:
            success_def = GCValue(
                hidden_dims=config['dynamics_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=1,
            )
            reward_def = GCValue(
                hidden_dims=config['dynamics_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=1,
            )
            network_info.update({
                'success': (success_def, (ex_observations, ex_goals, ex_actions)),
                'reward': (reward_def, (ex_observations, ex_goals, ex_actions)),
            })

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config['ob_dim'] = ob_dim
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='model_ac',  # Agent name.
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
            rollout_batch_size=1024,  # Batch size for model-based rollouts.
            rollout_steps=10,  # Model-based rollout steps.
            lam=1.0,  # GAE lambda.
            target_network=True,  # Whether to use target networks.
            value_loss='squared',  # Value loss type ('squared' or 'bce').
            flow_steps=10,  # Number of flow steps.
            num_samples_train=8,  # Number of samples for the actor.
            num_samples_test=32, # Number of samples for the actor.
            action_chunking=10,
            action_chunking_test=1,
            gc_actor=True,
            onestep_actor=True,
            learn_success=False,
            sarsa_backup=False,
            model_backup=False,
            rollout_critic=False,
            rollout_value=True,
            # Dataset hyperparameters.
            dataset_class='ACGCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=False,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.5,  # Probability of using a random state as the actor goal.
            actor_geom_sample=True,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
        )
    )
    return config

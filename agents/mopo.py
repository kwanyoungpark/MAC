import copy
from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Dynamics, GCValue, GCActor, LogParam

sg = lambda x: jax.lax.stop_gradient(x)

class MOPOAgent(flax.struct.PyTreeNode):
    """MOPO (Yu et al., 2020) agent."""

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
        pred_next_ob_dist = self.network.select('dynamics')(batch['observations'], batch['actions'], params=grad_params)
        dynamics_loss = -pred_next_ob_dist.log_prob(batch['next_observations'][None]).mean()
        pred_next_ob = pred_next_ob_dist.sample(seed=rng)
        dynamics_mse = jnp.square(pred_next_ob - batch['next_observations'][None]).mean()

        return dynamics_loss, {
            'dynamics_loss': dynamics_loss,
            'dynamics_mse': dynamics_mse,
        }

    def reward_loss(self, batch, grad_params, rng):
        """Compute the reward prediction loss."""
        goals = batch['value_goals'] if 'value_goals' in batch else None
        rewards = self.network.select('reward')(batch['observations'], goals, batch['actions'], params=grad_params)

        rewards_target = batch['rewards']
        reward_loss = jnp.square(rewards_target - rewards).mean()

        return reward_loss, {
            'reward_loss': reward_loss,
            'reward_mean': rewards.mean(),
            'reward_max': rewards.max(),
            'reward_min': rewards.min(),
            'data_reward_mean': batch['rewards'].mean(),
        }

    def success_loss(self, batch, grad_params, rng):
        """Compute the success prediction loss."""
        goals = batch['value_goals'] if 'value_goals' in batch else None
        succ_logit = self.network.select('success')(batch['observations'], goals, batch['actions'], params=grad_params)

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

    def actor_loss(self, batch, grad_params, rng):
    	"""Compute actor loss of MOPO (SAC with target entropy)"""
        dist = self.network.select('actor')(batch['observations'], goals=batch['actor_goals'], params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=rng)
       
        # SAC loss
        qs = self.network.select('critic')(batch['observations'], batch['actor_goals'], actions)
        q = jnp.mean(qs, axis=0)
        actor_loss = (log_probs * self.network.select('lam')() - q).mean()

        # Entropy loss.
        lam = self.network.select('lam')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        lam_loss = (lam * (sg(entropy) - self.config['target_entropy'])).mean()

        total_loss = actor_loss + lam_loss
        if self.config['tanh_squash']:
            action_std = dist._distribution.stddev()
        else:
            action_std = dist.stddev().mean()

        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'lam_loss': lam_loss,
            'lam': lam,
            'entropy': -log_probs.mean(),
            'std': action_std.mean(),
            'q': q.mean(),
        }

    def compute_penalty(self, observations, actions, rng):
        """ Calculate MOPO penalty. """
        next_observations_dist = self.network.select('dynamics')(observations, actions)
        stddev = next_observations_dist.stddev()
        penalty = jnp.linalg.norm(stddev, axis=-1).max(axis=0)
        return penalty

    def preprocess(self, real_batch, fake_batch, rng):
        penalty = self.compute_penalty(fake_batch['observations'], fake_batch['actions'], rng)
        batch = {
            'observations': jnp.concatenate([real_batch['observations'], fake_batch['observations']], axis=0),
            'actions': jnp.concatenate([real_batch['actions'], fake_batch['actions']], axis=0),
            'next_observations': jnp.concatenate([real_batch['next_observations'], fake_batch['next_observations']], axis=0),
            'rewards': jnp.concatenate([real_batch['rewards'], fake_batch['rewards'] - self.config['beta'] * penalty], axis=0),
            'masks': jnp.concatenate([real_batch['masks'], fake_batch['masks']], axis=0),
        }
        if 'value_goals' not in real_batch:
            batch.update({
                'value_goals': None,
                'actor_goals': None, 
            })
        else:
            batch.update({
                'value_goals': jnp.concatenate([real_batch['value_goals'], fake_batch['value_goals']], axis=0),
                'actor_goals': jnp.concatenate([real_batch['actor_goals'], fake_batch['actor_goals']], axis=0),
            })
        return batch, {
            'penalty': penalty.mean(),
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the SAC critic loss."""
        rng, sample_rng = jax.random.split(rng)
        next_dist = self.network.select('actor')(batch['next_observations'], goals=batch['value_goals'])
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=sample_rng)

        next_qs = self.network.select('target_critic')(
            batch['next_observations'], goals=batch['value_goals'], actions=next_actions
        )

        if self.config['value_loss'] == 'bce':
            next_qs = jax.nn.sigmoid(next_qs)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        elif self.config['q_agg'] == 'mean':
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        
        # Clip target Q values to the valid range.
        if batch['value_goals'] is not None:
            if self.config['gc_negative']:
                target_q = jnp.clip(target_q, -1 / (1 - self.config['discount']), 0)
            else:
                target_q = jnp.clip(target_q, 0, 1)
        if self.config['value_loss'] == 'squared':
            q = self.network.select('critic')(
                batch['observations'], goals=batch['value_goals'], actions=batch['actions'], params=grad_params
            )
            critic_loss = jnp.square(q - target_q).mean()

            return critic_loss, {
                'critic_loss': critic_loss,
                'q_mean': q.mean(),
                'q_max': q.max(),
                'q_min': q.min(),
            }
        elif self.config['value_loss'] == 'bce':
            q_logit = self.network.select('critic')(
                batch['observations'], goals=batch['value_goals'], actions=batch['actions'], params=grad_params
            )
            q = jax.nn.sigmoid(q_logit)
            log_q = jax.nn.log_sigmoid(q_logit)
            log_not_q = jax.nn.log_sigmoid(-q_logit)
            critic_loss = -(log_q * target_q + log_not_q * (1 - target_q)).mean()

            return critic_loss, {
                'critic_loss': critic_loss,
                'q_mean': q.mean(),
                'q_max': q.max(),
                'q_min': q.min(),
                'q_logit_mean': q_logit.mean(),
                'q_logit_max': q_logit.max(),
                'q_logit_min': q_logit.min(),
            }

    @jax.jit
    def rollout(self, batch, rng):
        """Perform model-based rollouts."""
        init_ob = batch['observations']
        if 'value_goals' in batch:
            value_goals = batch['value_goals']
            actor_goals = batch['actor_goals']
        else:
            value_goals = None
            actor_goals = None

        def rollout_step(carry, _):
            ob, rng = carry
            rng, action_rng, dyn_rng = jax.random.split(rng, 3)

            action, _ = self.sample_actions(ob, value_goals, action_rng, mode='train')
            next_ob = self.sample_dynamics(ob, action, dyn_rng)
            return (next_ob, rng), {
                'observations': ob,
                'actions': action,
                'next_observations': next_ob,
            }

        rng, rollout_rng = jax.random.split(rng)
        init_carry = (init_ob, rollout_rng)
        final_carry, rollout_data = jax.lax.scan(rollout_step, init_carry, None, length=self.config['rollout_steps'])

        if value_goals is not None:
            rollout_value_goals = jnp.repeat(batch['value_goals'][None], self.config['rollout_steps'], axis=0)
            rollout_actor_goals = jnp.repeat(batch['actor_goals'][None], self.config['rollout_steps'], axis=0)
            success = jax.nn.sigmoid(self.network.select('success')(rollout_data['observations'], rollout_value_goals, rollout_data['actions']))
            success = jnp.where(success > 0.5, jnp.ones_like(success), jnp.zeros_like(success))
            rewards = success - (1.0 if self.config['gc_negative'] else 0.0)
            rollout_data['masks'] = 1.0 - success

            new_batch = {
                'observations': jnp.concatenate(rollout_data['observations'], axis=0),
                'actions': jnp.concatenate(rollout_data['actions'], axis=0),
                'next_observations': jnp.concatenate(rollout_data['next_observations'], axis=0),
                'rewards': jnp.concatenate(rollout_data['rewards'], axis=0),
                'value_goals': rollout_value_goals, 
                'actor_goals': rollout_actor_goals, 
                'masks': jnp.concatenate(rollout_data['masks'], axis=0)
            }
        else:
            success = jax.nn.sigmoid(self.network.select('success')(rollout_data['observations'], None, rollout_data['actions'])) 
            success = jnp.where(success > 0.5, jnp.ones_like(success), jnp.zeros_like(success))
            rewards = self.network.select('reward')(rollout_data['observations'], None, rollout_data['actions'])
            new_batch = {
                'observations': jnp.concatenate(rollout_data['observations'], axis=0),
                'actions': jnp.concatenate(rollout_data['actions'], axis=0),
                'next_observations': jnp.concatenate(rollout_data['next_observations'], axis=0),
                'rewards': jnp.concatenate(rewards, axis=0),
                'masks': 1 - jnp.concatenate(success, axis=0), 
            }
        return new_batch
   
    @jax.jit
    def total_loss(self, real_batch, fake_batch, grad_params, rng=None):
        """Compute the total loss."""
        losses, info = {}, {}
        rng = rng if rng is not None else self.rng

        rng, dynamics_rng, actor_rng, preprocess_rng, critic_rng, success_rng = jax.random.split(rng, 6)

        dynamics_loss, dynamics_info = self.dynamics_loss(real_batch, grad_params, dynamics_rng)
        losses['dynamics'] = dynamics_loss
        for k, v in dynamics_info.items():
            info[f'dynamics/{k}'] = v

        success_loss, success_info = self.success_loss(real_batch, grad_params, success_rng)
        for k, v in success_info.items():
            info[f'success/{k}'] = v
        losses['success'] = success_loss

        reward_loss, reward_info = self.reward_loss(real_batch, grad_params, success_rng)
        for k, v in reward_info.items():
            info[f'reward/{k}'] = v
        losses['reward'] = reward_loss

        batch, preprocess_info = self.preprocess(real_batch, fake_batch, preprocess_rng)
        for k, v in preprocess_info.items():
            info[f'preprocess/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        losses['actor'] = actor_loss
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        losses['critic'] = critic_loss

        loss = sum(losses.values())
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
    def update(self, real_batch, fake_batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(real_batch, fake_batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def compute_metrics(self, real_batch, fake_batch, rng=None):
        goals = real_batch['actor_goals'] if 'actor_goals' in real_batch else None
        actions, _ = self.sample_actions(real_batch['observations'], goals, seed=rng, mode='train')
        mse = jnp.mean((actions - real_batch['actions']) ** 2)

        info = {
            'mse': mse,
        }

        return info

    def sample_dynamics(
        self,
        observations,
        actions,
        seed=None,
    ):
        """Sample next state from the world model."""
        dyn_rng, en_rng = jax.random.split(seed)
        next_obs_dist = self.network.select('dynamics')(observations, actions)
        next_obs = next_obs_dist.sample(seed=dyn_rng)
        idxs = jax.random.choice(en_rng, next_obs.shape[0], (next_obs.shape[1],))
        next_ob = next_obs[idxs, jnp.arange(next_obs.shape[1])]
        return next_ob

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
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions, {}
    
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
        if 'actor_goals' in example_batch: ex_goals = example_batch['actor_goals']
        else: ex_goals = None
        ex_times = ex_actions[..., :1]
        ob_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim

        # Define networks.
        dynamics_def = Dynamics(
            hidden_dims=config['dynamics_hidden_dims'],
            output_dim=ob_dim,
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_dynamics'],
            stochastic=True,
        )
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
            tanh_squash=config['tanh_squash'],
            state_dependent_std=config['state_dependent_std'],
            const_std=False,
            final_fc_init_scale=config['actor_fc_scale'],
        )
        lam_def = LogParam()
    
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

        network_info = dict(
            dynamics=(dynamics_def, (ex_observations, ex_actions)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
            success=(success_def, (ex_observations, ex_goals, ex_actions)),
            reward=(reward_def, (ex_observations, ex_goals, ex_actions)),
            lam=(lam_def, ()),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dim'] = ob_dim
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='mopo',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor network hidden dimensions.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Value network hidden dimensions.
            dynamics_hidden_dims=(1024, 1024, 1024, 1024),  # Dynamics network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.999,  # Discount factor.
            tau=0.005,  # Target network update rate.
            target_entropy=ml_collections.config_dict.placeholder(float),  # Target entropy (None for automatic tuning).
            target_entropy_multiplier=0.5,  # Multiplier to dim(A) for target entropy.
            tanh_squash=True,  # Whether to squash actions with tanh.
            state_dependent_std=True,  # Whether to use state-dependent standard deviations for actor.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            num_qs=2,  # Number of Q ensembles.
            num_dynamics=7,
            q_agg='min',  # Aggregation function for Q values.
            ob_dim=ml_collections.config_dict.placeholder(int),  # Observation dimension (set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (set automatically).
            target_network=True,  # Whether to use target networks.
            value_loss='squared',  # Value loss type ('squared' or 'bce').
            num_samples=10,  # Number of samples for the actor.
            rollout_steps=5,
            beta=0.003,
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
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

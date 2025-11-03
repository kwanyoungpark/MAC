import copy
from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Dynamics, ActorVectorField, GCValue, RunningMeanStd

sg = jax.lax.stop_gradient
class MBFQLAgent(flax.struct.PyTreeNode):
    """MAC, but replacing rejection sampling to FQL."""

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
        pred_next_ob = self.network.select('dynamics')(batch['observations'], batch['actions'], params=grad_params)

        dynamics_loss = jnp.square(pred_next_ob - batch['next_observations']).mean()

        return dynamics_loss, {
            'dynamics_loss': dynamics_loss,
        }

    def reward_loss(self, batch, grad_params, rng):
        """Compute the reward prediction loss."""
        goals = batch['value_goals'] 
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

        actor_goals = batch['actor_goals']
        pred = self.network.select('actor_bc_flow')(
            batch['observations'], goals=actor_goals, actions=x_t, times=t, params=grad_params
        )
        flow_loss = jnp.mean((pred - vel) ** 2)

        return flow_loss, {
            'flow_loss': flow_loss,
            'vel': jnp.linalg.norm(vel, axis=-1).mean(),
        }
    
    def rollout(self, batch, grad_params, rng):
        """Perform model-based rollouts."""
        init_ob = batch['observations']
        goal = batch['value_goals']

        def rollout_step(carry, _):
            ob, rng = carry
            rng, action_rng = jax.random.split(rng, 2)

            action, _ = self.sample_actions(ob, goal, action_rng, mode='train')
            clipped_action = jnp.clip(action, -1.0, 1.0)
            next_ob = self.network.select('dynamics')(ob, clipped_action)

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
        batch['rollout_goals'] = jnp.repeat(goal[None], length, axis=0) if goal is not None else None

        success = jax.nn.sigmoid(self.network.select('success')(batch['rollout_observations'], batch['rollout_goals'], batch['rollout_actions']))
        success = jnp.where(success > 0.5, jnp.ones_like(success), jnp.zeros_like(success))
        rewards = self.network.select('reward')(batch['rollout_observations'], batch['rollout_goals'], batch['rollout_actions'])

        if goal is None:
            batch['rollout_rewards'] = rewards
        else:
            batch['rollout_rewards'] = success - (1.0 if self.config['gc_negative'] else 0.0)
        batch['rollout_masks'] = 1.0 - success

    def compute_gae(self, batch):
        """Compute GAE returns and normalized advantages."""
        # batch: dict of (num_steps, num_envs, ob_dim)
        value_module = 'target_value' if self.config['target_network'] else 'value'
        values = self.network.select(value_module)(batch['rollout_observations'], batch['rollout_goals'])
        next_values = self.network.select(value_module)(batch['rollout_next_observations'], batch['rollout_goals'])

        if self.config['value_loss'] == 'bce':
            values = jax.nn.sigmoid(values)
            next_values = jax.nn.sigmoid(next_values)

        def scan_fn(lastgaelam, inputs):
            reward, mask, value, next_value = inputs
            delta = reward + mask * self.config['discount'] * next_value - value
            advantage = delta + mask * self.config['discount'] * self.config['lam'] * lastgaelam
            return advantage, advantage

        zeros = jnp.zeros(batch['rollout_rewards'].shape[1])
        _, advs = jax.lax.scan(
            scan_fn, zeros, (batch['rollout_rewards'], batch['rollout_masks'], values, next_values), reverse=True
        )
        returns = values + advs

        batch['rollout_returns'] = returns

        return {
            'return_mean': returns.mean(),
            'return_min': returns.min(),
            'return_max': returns.max(),
            'return_std': returns.std(),
        }

    def value_loss(self, batch, grad_params, rng):
        """Compute the value loss."""
        v = self.network.select('value')(batch['rollout_observations'], batch['rollout_goals'], params=grad_params)
        if self.config['value_loss'] == 'squared':
            value_loss = ((v - batch['rollout_returns']) ** 2).mean()
        elif self.config['value_loss'] == 'bce':
            # Clip targets to [0, 1] for BCE loss.
            v_logit = v
            v = jax.nn.sigmoid(v_logit)
            target = jnp.clip(batch['rollout_returns'], 0, 1)
            value_loss = (self.bce_loss(v_logit, target)).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss."""
        rng, sample_rng = jax.random.split(rng, 2)
        batch_size, action_dim = batch['actions'].shape
       
        goal = batch['rollout_goals'][0] if batch['rollout_goals'] is not None else None
        q = self.network.select('critic')(batch['rollout_observations'][0], goal, batch['rollout_actions'][0], params=grad_params)

        discount = self.config['discount'] ** self.config['action_chunking'] 
        reward = batch['rollout_rewards'][0]
        mask = batch['rollout_masks'][0]
        next_v = self.network.select('value')(batch['rollout_next_observations'][0], goal)
        target = sg(reward + discount * mask * next_v)

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

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, batch['rollout_actions'].shape)
        target_flow_actions = self.sample_flow_actions(batch['rollout_observations'], goals=batch['rollout_goals'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(
            batch['rollout_observations'], goals=batch['rollout_goals'], actions=noises, params=grad_params
        )
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['rollout_observations'], goals=batch['rollout_goals'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        # Make it scale-invariant.
        q_loss = -q.mean()
        lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
        q_loss = lam * q_loss

        # Total loss.
        actor_loss = self.config['alpha'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
        }


    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        losses = {}
        rng = rng if rng is not None else self.rng

        rng, dynamics_rng, bc_actor_rng, success_rng, rollout_rng, critic_rng, value_rng, actor_rng = jax.random.split(rng, 8)

        dynamics_loss, dynamics_info = self.dynamics_loss(batch, grad_params, dynamics_rng)
        for k, v in dynamics_info.items():
            info[f'dynamics/{k}'] = v
        losses['dynamics'] = dynamics_loss

        success_loss, success_info = self.success_loss(batch, grad_params, success_rng)
        for k, v in success_info.items():
            info[f'success/{k}'] = v
        losses['success'] = success_loss

        reward_loss, reward_info = self.reward_loss(batch, grad_params, success_rng)
        for k, v in reward_info.items():
            info[f'reward/{k}'] = v
        losses['reward'] = reward_loss

        bc_actor_loss, bc_actor_info = self.bc_actor_loss(batch, grad_params, bc_actor_rng)
        for k, v in bc_actor_info.items():
            info[f'bc_actor/{k}'] = v
        losses['bc_actor'] = bc_actor_loss

        self.rollout(batch, grad_params, rollout_rng)
        gae_info = self.compute_gae(batch)
        for k, v in gae_info.items():
            info[f'gae/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        losses['critic'] = critic_loss

        value_loss, value_info = self.value_loss(batch, grad_params, value_rng)
        for k, v in value_info.items():
            info[f'value/{k}'] = v
        losses['value'] = value_loss

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        losses['actor'] = actor_loss
        
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
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')

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
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        actions = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor_onestep_flow')(observations=observations, goals=goals, actions=actions)
        actions = jnp.clip(actions, -1, 1)
        if mode == 'test':
            action_dim = actions.shape[-1] // self.config['action_chunking']
            actions = actions[..., :action_dim * self.config['action_chunking_test']]
        return actions, {}
   
    @jax.jit
    def sample_flow_actions(
        self,
        observations,
        noises,
        goals=None,
    ):
        """Sample actions from the BC flow policy."""
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, goals=goals, actions=actions, times=t)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

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
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define networks.
        dynamics_def = Dynamics(
            hidden_dims=config['dynamics_hidden_dims'],
            output_dim=ob_dims[-1],
            layer_norm=config['layer_norm'],
        )
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
        )
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
        )

        network_info = dict(
            dynamics=(dynamics_def, (ex_observations, ex_actions)),
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goals)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_goals, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_goals, ex_actions, None)),
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

        params = network.params
        params['modules_target_value'] = params['modules_value']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='mbfql',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor network hidden dimensions.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Value network hidden dimensions.
            dynamics_hidden_dims=(1024, 1024, 1024, 1024),  # Dynamics network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.999,  # Discount factor.
            tau=0.005,  # Target network update rate.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (set automatically).
            flow_steps=10,  # Number of flow steps.
            rollout_batch_size=1024,  # Batch size for model-based rollouts.
            rollout_steps=10,  # Model-based rollout steps.
            lam=1.0,  # GAE lambda.
            target_network=True,  # Whether to use target networks.
            adv_normalization='none',  # Advantage normalization type ('none', 'adv', 'return').
            ent_coef=3e-4,  # Entropy coefficient.
            bc_loss='mode_seeking',  # BC loss type ('mode_seeking' or 'mode_covering').
            alpha=0.3,  # BC coefficient.
            value_loss='squared',  # Value loss type ('squared' or 'bce').
            # Dataset hyperparameters.
            dataset_class='ACGCDataset',  # Dataset class name.
            action_chunking=5,
            action_chunking_test=1,
            learn_success=True,
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

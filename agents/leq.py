import copy
from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Dynamics, GCDeterActor, GCValue, RunningMeanStd

def expectile_loss(target, pred, expectile):
    weight = jnp.where(target > pred, expectile, (1 - expectile))
    diff = target - pred
    return weight * (diff**2)

class LEQAgent(flax.struct.PyTreeNode):
    """LEQ (Park et al., 2024) agent."""

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

    def sample_reward(self, s, g, a):
        """Compute the reward prediction loss."""
        success = jax.nn.sigmoid(self.network.select('success')(s, g, a))
        success = jnp.where(success > 0.5, jnp.ones_like(success), jnp.zeros_like(success))
        rewards = self.network.select('reward')(s, g, a)
        if g is None:
            return rewards, success 
        return success - 1.0, success

    def critic_loss(self, batch, grad_params, rng):
    	"""Compute the critic loss of LEQ (mostly copied from the original implementation)"""
        rng, keys = jax.random.split(rng)
        expectile = self.config['expectile']
        num_repeat = self.config['num_repeat']
        lamb = self.config['lam']
        discount = self.config['discount']
        H = self.config['rollout_steps']
        N = batch['observations'].shape[0]
        model_batch_ratio = self.config['model_batch_ratio']
        key1, key2, key3, key4 = jax.random.split(rng, 4)

        ## Generate imaginary trajectories
        Robs = (
            batch['observations'][:, None, :]
            .repeat(repeats=num_repeat, axis=1)
            .reshape(N * num_repeat, -1)
        )
        if 'value_goals' not in batch: 
            Rg = None
            use_gc = False
            value_goals = None
        else:
            Rg = (
                batch['value_goals'][:, None, :]
                .repeat(repeats=num_repeat, axis=1)
                .reshape(N * num_repeat, -1)
            )
            use_gc = True
            value_goals = batch['value_goals']

        Ra = self.network.select('actor')(Robs, Rg)
        states, goals, rewards, actions, masks, mask_weights, loss_weights = (
            [Robs],
            [Rg],
            [],
            [Ra],
            [],
            [jnp.ones(N * num_repeat)],
            [jnp.ones(N * num_repeat)],
        )
        for i in range(H):
            s, a = states[-1], actions[-1]
            rng1, rng2, key1 = jax.random.split(key1, 3)
            s_next = self.sample_dynamics(s, a, rng1)

            rew, terminal = self.sample_reward(states[i], goals[i], actions[i])
            a_next = self.network.select('actor')(s_next, Rg)
            states.append(s_next)
            goals.append(Rg)
            actions.append(a_next)
            rewards.append(rew)
            masks.append(1 - terminal)
            mask_weights.append(mask_weights[i] * (1 - terminal))
            loss_weights.append(loss_weights[i] * (1 - terminal) * discount)

        mask_weights = jnp.stack(mask_weights, axis=0)
        loss_weights = jnp.stack(loss_weights[:-1], axis=0)

        ## Calculate lambda-returns
        target_q_rollout, lamb_weight = [self.network.select('critic')(states[-1], Rg, actions[-1])], 1.0
        for i in reversed(range(H)):
            q_cur = self.network.select('critic')(states[i], Rg, actions[i])
            q_next = (
                mask_weights[i] * rewards[i]
                + mask_weights[i + 1] * discount * target_q_rollout[-1]
            )
            next_value = (q_cur + lamb * lamb_weight * q_next) / (1 + lamb * lamb_weight)
            target_q_rollout.append(next_value)
            lamb_weight = 1.0 + lamb * lamb_weight
        target_q_rollout = list(reversed(target_q_rollout))[:-1]

        target_q_rollout = jnp.stack(target_q_rollout, axis=0)
        states = jnp.stack(states[:-1], axis=0)
        goals = jnp.stack(goals[:-1], axis=0) if use_gc else None
        actions = jnp.stack(actions[:-1], axis=0)
        rewards = jnp.stack(rewards, axis=0)

        ## Calculate target for dataset transitions
        next_a = self.network.select('actor')(batch['next_observations'], Rg)
        next_value = self.network.select('critic')(batch['next_observations'], Rg, next_a)
        target_q_data = batch['rewards'] + discount * batch['masks'] * next_value

        # Calculate critic loss for dataset transitions
        q_data = self.network.select('critic')(batch['observations'], value_goals, batch['actions'], params=grad_params) 
        critic_loss_data = expectile_loss(target_q_data, q_data, 0.5).mean()

        # Calculate critic loss for imaginated trajectories
        q_rollout = self.network.select('critic')(states, goals, actions)
        critic_loss_rollout = expectile_loss(target_q_rollout, q_rollout, expectile)
        critic_loss_rollout = (critic_loss_rollout * loss_weights).mean()

        # EMA regularization loss
        q_target_data = self.network.select('target_critic')(batch['observations'], value_goals, batch['actions'])
        critic_reg_loss_data = jnp.mean((q_target_data - q_data) ** 2)
        q_target_rollout = self.network.select('target_critic')(states, goals, actions)
        critic_reg_loss_rollout = (q_target_rollout - q_rollout) ** 2
        critic_reg_loss_rollout = jnp.mean(critic_reg_loss_rollout * loss_weights)

        critic_reg_loss = (
            critic_reg_loss_rollout * model_batch_ratio
            + critic_reg_loss_data * (1 - model_batch_ratio)
        )
        return critic_loss_data * (1 - model_batch_ratio) + critic_loss_rollout * model_batch_ratio + critic_reg_loss, {
            "critic_loss_data": critic_loss_data.mean(),
            "critic_loss_model": critic_loss_rollout.mean(),
            "q_data": q_data.mean(),
            "q_data_min": q_data.min(),
            "q_data_max": q_data.max(),
            "reward_data": batch['rewards'].mean(),
            "reward_data_max": batch['rewards'].max(),
            "reward_data_min": batch['rewards'].min(),
            "q_model": q_rollout.mean(),
            "q_model_min": q_rollout.min(),
            "q_model_max": q_rollout.max(),
            "reward_model": (rewards * mask_weights[:-1]).mean(),
            "reward_max": (rewards * mask_weights[:-1]).max(),
            "reward_min": (rewards * mask_weights[:-1]).min(),
            "critic_reg_loss": critic_reg_loss,
            "critic_reg_loss_data": critic_reg_loss_data,
            "critic_reg_loss_rollout": critic_reg_loss_rollout,
            "mask_weights": mask_weights.mean(),
            "loss_weights": loss_weights.mean(),
            "expectile": expectile,
            "model_batch_ratio": model_batch_ratio,
            "state_max": jnp.abs(states * mask_weights[:-1, :, None]).max(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss of LEQ (mostly copied from the original implementation)"""
        rng, key = jax.random.split(rng)
        expectile = self.config['expectile']
        num_repeat = self.config['num_repeat']
        lamb = self.config['lam']
        discount = self.config['discount']
        H = self.config['rollout_steps']

        N = batch['observations'].shape[0]
        Robs = (
            batch['observations'][:, None, :]
            .repeat(repeats=num_repeat, axis=1)
            .reshape(N * num_repeat, -1)
        )
        if 'actor_goals' not in batch: 
            Rg = None
            use_gc = False
        else:
            Rg = (
                batch['actor_goals'][:, None, :]
                .repeat(repeats=num_repeat, axis=1)
                .reshape(N * num_repeat, -1)
            )
            use_gc = True

        Ra = self.network.select('actor')(Robs, Rg)

        def calculate_gae_foward(Robs, Rg, Ra, key0):
            ## Generate imagined trajectory
            states, goals, rewards, actions, mask_weights, keys = [Robs], [Rg], [], [Ra], [1.0], [key0]
            q_rollout, q_values, ep_weights = [], [self.network.select('critic')(Robs, Rg, Ra)], []
            for i in range(H):
                rng1, rng2, rng3, key0 = jax.random.split(keys[-1], 4)
                keys.append(key0)
                s_next = self.sample_dynamics(states[i], actions[i], rng1)
                rew, terminal = self.sample_reward(states[i], goals[i], actions[i])
                a_next = self.network.select('actor')(s_next, Rg)
                states.append(s_next)
                goals.append(Rg)
                actions.append(a_next)
                rewards.append(rew)
                mask_weights.append(mask_weights[i] * (1 - terminal))
                q_values.append(self.network.select('critic')(s_next, Rg, a_next))

            ## Calculate lambda-returns
            q_rollout, lamb_weight = [q_values[-1]], 1.0
            for i in reversed(range(H)):
                q_next = (
                    mask_weights[i] * rewards[i]
                    + mask_weights[i + 1] * discount * q_rollout[-1]
                )
                next_value = (q_values[i] + lamb * lamb_weight * q_next) / (
                    1 + lamb * lamb_weight
                )
                q_rollout.append(next_value)
                lamb_weight = 1.0 + lamb * lamb_weight
            q_rollout = list(reversed(q_rollout))

            ## Calculate asymmetric weights
            ep_weights = []
            for i in range(H):
                ep_weights.append(
                    jnp.where(q_rollout[i] > q_values[i], expectile, 1 - expectile)
                )
            ep_weights.append(0.5)

            states = jnp.stack(states, axis=0)
            goals = jnp.stack(goals, axis=0) if Rg is not None else None 
            actions = jnp.stack(actions, axis=0)
            mask_weights = jnp.stack(mask_weights, axis=0)
            q_rollout = jnp.stack(q_rollout, axis=0)
            ep_weights = jnp.stack(ep_weights, axis=0)
            return states, goals, actions, mask_weights, q_rollout, ep_weights

        keys = jax.random.split(key, N * num_repeat)
        vmap_foward = lambda func: jax.vmap(func, in_axes=0, out_axes=1)
        states0, goals0, actions0, mask_weights0, q_rollout, ep_weights = vmap_foward(
            calculate_gae_foward
        )(Robs, Rg, Ra, keys)

        def calculate_gae_backward(delta_a, Robs, Rg, Ra, key0):
            ## Generate imagined trajectory (identical with foward)
            states, rewards, actions, mask_weights, keys = (
                [Robs],
                [],
                [Ra + delta_a[0]],
                [1.0],
                [key0],
            )
            q_rollout, q_values, ep_weights = [], [self.network.select('critic')(Robs, Rg, Ra + delta_a[0])], []
            for i in range(H):
                rng1, rng2, rng3, key0 = jax.random.split(keys[-1], 4)
                keys.append(key0)
                s_next = self.sample_dynamics(states[i], actions[i], rng1)

                rew, terminal = self.sample_reward(states[i], Rg, actions[i])
                a_next = self.network.select('actor')(s_next, Rg) + delta_a[i + 1]
                states.append(s_next)
                actions.append(a_next)
                rewards.append(rew)
                mask_weights.append(mask_weights[i] * (1 - terminal))
                q_values.append(self.network.select('critic')(s_next, Rg, a_next))

            ## Calculate lambda-returns
            q_rollout, lamb_weight = [q_values[-1]], 1.0
            for i in reversed(range(H)):
                q_next = (
                    mask_weights[i] * rewards[i]
                    + mask_weights[i + 1] * discount * q_rollout[-1]
                )
                next_value = (q_values[i] + lamb * lamb_weight * q_next) / (
                    1 + lamb * lamb_weight
                )
                q_rollout.append(next_value)
                lamb_weight = 1.0 + lamb * lamb_weight
            q_rollout = list(reversed(q_rollout))

            return jnp.stack(q_rollout, axis=0)

        ## Calculate gradient of Q_t^{\lambda} w.r.t a_t
        delta_a = jnp.zeros_like(actions0)
        vmap_backward = lambda func: jax.vmap(func, in_axes=(1, 0, 0, 0, 0), out_axes=1)
        grad_gae = vmap_backward(jax.jacrev(calculate_gae_backward))(
            delta_a, Robs, Rg, Ra, keys
        )
        grad_gae = jnp.stack([grad_gae[i, :, i] for i in range(H + 1)])


        # Finally the loss function
        actions_grad = self.network.select('actor')(states0, goals0, params=grad_params)

        ## Calculate gradient of Q_t^{\lambda} w.r.t parameter using deterministic policy gradient theorem (chain rule)
        actor_loss = (
            -(ep_weights[:, :, None] * grad_gae * actions_grad).mean(axis=1).sum()
        )

        return actor_loss, {
            "actor_loss": actor_loss,
            "q_rollout": q_rollout.mean(),
            "lambda_actor": lamb,
            "adv_weights": (ep_weights * mask_weights0).mean() / mask_weights0.mean(),
            "abs_actions": jnp.abs(actions0).mean(),
        }


    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, dynamics_rng, critic_rng, actor_rng, success_rng = jax.random.split(rng, 5)

        losses = {}

        dynamics_loss, dynamics_info = self.dynamics_loss(batch, grad_params, dynamics_rng)
        for k, v in dynamics_info.items():
            info[f'dynamics/{k}'] = v
        losses['dynamics'] = dynamics_loss

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        losses['critic'] = critic_loss

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        losses['actor'] = actor_loss

        success_loss, success_info = self.success_loss(batch, grad_params, success_rng)
        for k, v in success_info.items():
            info[f'success/{k}'] = v
        losses['success'] = success_loss

        reward_loss, reward_info = self.reward_loss(batch, grad_params, success_rng)
        for k, v in reward_info.items():
            info[f'reward/{k}'] = v
        losses['reward'] = reward_loss

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
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def compute_metrics(self, batch, rng=None):
        goals = batch['actor_goals'] if 'actor_goals' in batch else None
        actions, _ = self.sample_actions(batch['observations'], goals, seed=rng, mode='train')
        mse = jnp.mean((actions - batch['actions']) ** 2)

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
        """Sample next state from the world model"""
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
        actions = self.network.select('actor')(observations, goals)
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
        ob_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]

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
            num_ensembles=1,
        )
        actor_def = GCDeterActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
            tanh_squash=config['tanh_squash'],
            final_fc_init_scale=config['actor_fc_scale'],
        )
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
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='leq',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor network hidden dimensions.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Value network hidden dimensions.
            dynamics_hidden_dims=(1024, 1024, 1024, 1024),  # Dynamics network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.999,  # Discount factor.
            tau=0.005,  # Target network update rate.
            tanh_squash=True,  # Whether to squash actions with tanh.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            rollout_steps=10,  # Model-based rollout steps.
            lam=1.0,  # GAE lambda.
            value_loss='squared',  # Value loss type ('squared' or 'bce').
            expectile=0.5,
            num_repeat=1,
            model_batch_ratio=0.95,
            num_dynamics=7,
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

from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
from .datasets import normalize

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    env_name=None,
    goal_conditioned=True,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    scale=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        env_name: Environment name.
        goal_conditioned: Whether to do goal-conditioned evaluation.
        task_id: Task ID to be passed to the environment (only used when goal_conditioned is True).
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        score = 0.
        if goal_conditioned:
            observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
            goal = info.get('goal')
            goal_frame = info.get('goal_rendered')
        else:
            observation, info = env.reset()
            goal = None
            goal_frame = None

        done = False
        step = 0
        render = []
        action_chunks = []
        while not done:
            if len(action_chunks) == 0:
                observation_norm = normalize(scale['observations'], observation)
                goal_norm = None if goal is None else normalize(scale['oracle_reps'], goal)
                action, action_info = actor_fn(observations=observation_norm, goals=goal_norm, temperature=eval_temperature)
                action = np.array(action)
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)
                action_dim = env.action_space.shape[-1]
                action_chunks = []
                for j in range(0, action.shape[-1], action_dim):
                    action_chunks.append(action[j:j+action_dim])
           
            action = action_chunks.pop(0)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observations=normalize(scale['observations'], observation),
                next_observations=normalize(scale['observations'], next_observation),
                actions=action,
                rewards=reward,
                dones=done,
                value_goals=goal,
                **action_info,
            )
            add_to(traj, transition)
            observation = next_observation.copy()

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
            stats['score'].append(score) 
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders

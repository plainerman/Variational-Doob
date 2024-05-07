import jax
import jax.numpy as jnp
from tqdm import tqdm

MAX_STEPS = 1_000


class FirstOrderSystem:
    def __init__(self, start_state, target_state, step):
        self.start_state = start_state
        self.target_state = target_state
        self.step = step


def one_way_shooting(system, trajectory, fixed_length, key):
    key = jax.random.split(key)

    # pick a random point along the trajectory
    point_idx = jax.random.randint(key[0], (1,), 1, len(trajectory) - 1)[0]
    # pick a random direction, either forward or backward
    direction = jax.random.randint(key[1], (1,), 0, 2)[0]

    if direction == 0:
        trajectory = trajectory[:point_idx + 1]

    if direction == 1:
        trajectory = trajectory[point_idx:][::-1]

    steps = MAX_STEPS if fixed_length == 0 else fixed_length

    key, iter_key = jax.random.split(key[2])
    while len(trajectory) < steps:
        key, iter_key = jax.random.split(key)
        point = system.step(trajectory[-1], iter_key)
        trajectory.append(point)

        if jnp.isnan(point).any():
            return False, trajectory

        if system.start_state(trajectory[0]) and system.target_state(trajectory[-1]):
            if fixed_length == 0 or len(trajectory) == fixed_length:
                return True, trajectory
            return False, trajectory

        if system.target_state(trajectory[0]) and system.start_state(trajectory[-1]):
            if fixed_length == 0 or len(trajectory) == fixed_length:
                return True, trajectory[::-1]
            return False, trajectory

    return False, trajectory


def two_way_shooting(system, trajectory, fixed_length, key):
    key = jax.random.split(key)

    # pick a random point along the trajectory
    point_idx = jax.random.randint(key[0], (1,), 1, len(trajectory) - 1)[0]
    point = trajectory[point_idx]
    # simulate forward from the point until max_steps

    steps = MAX_STEPS if fixed_length == 0 else fixed_length

    key, iter_key = jax.random.split(key[1])
    new_trajectory = [point]
    while len(new_trajectory) < steps:
        key, iter_key = jax.random.split(key)
        point = system.step(new_trajectory[-1], iter_key)
        new_trajectory.append(point)

        if jnp.isnan(point).any():
            return False, trajectory

        if system.start_state(point) or system.target_state(point):
            break

    while len(new_trajectory) < steps:
        key, iter_key = jax.random.split(key)
        point = system.step(new_trajectory[0], iter_key)
        new_trajectory.insert(0, point)

        if jnp.isnan(point).any():
            return False, trajectory

        if system.start_state(point) or system.target_state(point):
            break

    # throw away the trajectory if it's not the right length
    if fixed_length != 0 and len(new_trajectory) != fixed_length:
        return False, trajectory

    if system.start_state(new_trajectory[0]) and system.target_state(new_trajectory[-1]):
        return True, new_trajectory

    if system.target_state(new_trajectory[0]) and system.start_state(new_trajectory[-1]):
        return True, new_trajectory[::-1]

    return False, trajectory


def mcmc_shooting(system, proposal, initial_trajectory, num_paths, key, fixed_length=0, warmup=50):
    # pick an initial trajectory
    trajectories = [initial_trajectory]

    with tqdm(total=num_paths) as pbar:
        while len(trajectories) <= num_paths + warmup:
            key, iter_key, accept_key = jax.random.split(key, 3)
            found, new_trajectory = proposal(system, trajectories[-1], fixed_length, iter_key)
            if not found:
                continue

            ratio = len(trajectories[-1]) / len(new_trajectory)
            # The first trajectory might have a very unreasonable length, so we skip it
            if len(trajectories) == 1 or jax.random.uniform(accept_key, shape=(1,)) < ratio:
                trajectories.append(new_trajectory)

                if len(trajectories) > warmup:
                    pbar.update(1)

    return trajectories[warmup + 1:]


def unguided_md(system, initial_point, num_paths, key, fixed_length=0):
    trajectories = []
    current_frame = initial_point.clone()
    current_trajectory = []

    with tqdm(total=num_paths) as pbar:
        while len(trajectories) < num_paths:
            key, iter_key = jax.random.split(key)
            next_frame = system.step(current_frame, iter_key)

            is_transition = not (system.start_state(next_frame) or system.target_state(next_frame))
            if is_transition:
                if len(current_trajectory) == 0:
                    current_trajectory.append(current_frame)

                if fixed_length != 0 and len(current_trajectory) > fixed_length:
                    current_trajectory = []
                    is_transition = False
                else:
                    current_trajectory.append(next_frame)
            elif len(current_trajectory) > 0:
                current_trajectory.append(next_frame)

                if fixed_length == 0 or len(current_trajectory) == fixed_length:
                    if system.start_state(current_trajectory[0]) and system.target_state(current_trajectory[-1]):
                        trajectories.append(current_trajectory)
                        pbar.update(1)
                    elif system.target_state(current_trajectory[0]) and system.start_state(current_trajectory[-1]):
                        trajectories.append(current_trajectory[::-1])
                        pbar.update(1)
                current_trajectory = []

            current_frame = next_frame

    return trajectories

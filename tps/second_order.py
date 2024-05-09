import jax
import jax.numpy as jnp
from tqdm import tqdm

MAX_STEPS = 2_000
MAX_ABS_VALUE = 5


class SecondOrderSystem:
    def __init__(self, start_state, target_state, step_forward, step_backward, sample_velocity):
        self.start_state = start_state
        self.target_state = target_state
        self.step_forward = step_forward
        self.step_backward = step_backward
        self.sample_velocity = sample_velocity


def one_way_shooting(system, trajectory, fixed_length, key):
    key = jax.random.split(key)

    # pick a random point along the trajectory
    point_idx = jax.random.randint(key[0], (1,), 1, len(trajectory) - 1)[0]
    # pick a random direction, either forward or backward
    direction = jax.random.randint(key[1], (1,), 0, 2)[0]

    # TODO: Fix correct dt in ps / pass previous velocities
    new_velocities = [(trajectory[point_idx] - trajectory[point_idx - 1]) / 0.001]

    if direction == 0:
        trajectory = trajectory[:point_idx + 1]
        step_function = system.step_forward
    else:  # direction == 1:
        trajectory = trajectory[point_idx:][::-1]
        step_function = system.step_backward

    steps = MAX_STEPS if fixed_length == 0 else fixed_length

    key, iter_key = jax.random.split(key[3])
    while len(trajectory) < steps:
        key, iter_key = jax.random.split(key)
        point, velocity = step_function(trajectory[-1], new_velocities[-1], iter_key)
        trajectory.append(point)
        new_velocities.append(velocity)

        if jnp.isnan(point).any() or jnp.isnan(velocity).any():
            return False, trajectory, new_velocities

        # ensure that our trajectory does not explode
        if (jnp.abs(point) > MAX_ABS_VALUE).any():
            return False, trajectory, new_velocities

        if system.start_state(trajectory[0]) and system.target_state(trajectory[-1]):
            if fixed_length == 0 or len(trajectory) == fixed_length:
                return True, trajectory, new_velocities
            return False, trajectory, new_velocities

        if system.target_state(trajectory[0]) and system.start_state(trajectory[-1]):
            if fixed_length == 0 or len(trajectory) == fixed_length:
                return True, trajectory[::-1], new_velocities[::-1]
            return False, trajectory, new_velocities

    return False, trajectory, new_velocities


def two_way_shooting(system, trajectory, fixed_length, key):
    key = jax.random.split(key)

    # pick a random point along the trajectory
    point_idx = jax.random.randint(key[0], (1,), 1, len(trajectory) - 1)[0]
    point = trajectory[point_idx]
    # simulate forward from the point until max_steps

    steps = MAX_STEPS if fixed_length == 0 else fixed_length

    new_trajectory = [point]
    new_velocities = [system.sample_velocity(key[1])]

    key, iter_key = jax.random.split(key[2])
    while len(new_trajectory) < steps:
        key, iter_key = jax.random.split(key)
        point, velocity = system.step_forward(new_trajectory[-1], new_velocities[-1], iter_key)
        new_trajectory.append(point)
        new_velocities.append(velocity)

        if jnp.isnan(point).any() or jnp.isnan(velocity).any():
            return False, new_trajectory, new_velocities

        # ensure that our trajectory does not explode
        if (jnp.abs(point) > MAX_ABS_VALUE).any():
            return False, new_trajectory, new_velocities

        if system.start_state(point) or system.target_state(point):
            break

    while len(new_trajectory) < steps:
        key, iter_key = jax.random.split(key)
        point, velocity = system.step_backward(new_trajectory[0], new_velocities[0], iter_key)
        new_trajectory.insert(0, point)
        new_velocities.insert(0, velocity)

        if jnp.isnan(point).any() or jnp.isnan(velocity).any():
            return False, new_trajectory, new_velocities

        # ensure that our trajectory does not explode
        if (jnp.abs(point) > MAX_ABS_VALUE).any():
            return False, new_trajectory, new_velocities

        if system.start_state(point) or system.target_state(point):
            break

    # throw away the trajectory if it's not the right length
    if fixed_length != 0 and len(new_trajectory) != fixed_length:
        return False, new_trajectory, new_velocities

    if system.start_state(new_trajectory[0]) and system.target_state(new_trajectory[-1]):
        return True, new_trajectory, new_velocities

    if system.target_state(new_trajectory[0]) and system.start_state(new_trajectory[-1]):
        return True, new_trajectory[::-1], new_velocities[::-1]

    return False, new_trajectory, new_velocities


def mcmc_shooting(system, proposal, initial_trajectory, num_paths, key, fixed_length=0, warmup=50):
    # pick an initial trajectory
    trajectories = [initial_trajectory]
    velocities = []
    statistics = {
        'num_force_evaluations': 0,
        'num_tries': 0,
        'num_metropolis_rejected': 0,
        'warmup': warmup,
        'num_paths': num_paths,
        'max_steps': MAX_STEPS,
        'max_abs_value': MAX_ABS_VALUE,
    }
    if fixed_length > 0:
        statistics['fixed_length'] = fixed_length

    with tqdm(total=num_paths + warmup, desc='warming up' if warmup > 0 else '') as pbar:
        while len(trajectories) <= num_paths + warmup:
            statistics['num_tries'] += 1
            if len(trajectories) > warmup:
                pbar.set_description('')

            key, traj_idx_key, iter_key, accept_key = jax.random.split(key, 4)
            traj_idx = jax.random.randint(traj_idx_key, (1,), warmup + 1, len(trajectories))[0]
            # during warmup, we want an iterative scheme
            traj_idx = traj_idx if traj_idx < len(trajectories) else -1

            found, new_trajectory, new_velocities = proposal(system, trajectories[traj_idx], fixed_length, iter_key)
            statistics['num_force_evaluations'] += len(new_trajectory) - 1

            if not found:
                continue

            ratio = len(trajectories[-1]) / len(new_trajectory)
            # The first trajectory might have a very unreasonable length, so we skip it
            if len(trajectories) == 1 or jax.random.uniform(accept_key, shape=(1,)) < ratio:
                trajectories.append(new_trajectory)
                velocities.append(new_velocities)
                pbar.update(1)
            else:
                statistics['num_metropolis_rejected'] += 1

    return trajectories[warmup + 1:], velocities[warmup:], statistics


def unguided_md(system, initial_point, num_paths, key, fixed_length=0):
    trajectories = []
    current_frame = initial_point.clone()
    current_trajectory = []

    key, velocity_key = jax.random.split(key)
    velocity = system.sample_velocity(velocity_key)

    with tqdm(total=num_paths) as pbar:
        while len(trajectories) < num_paths:
            key, iter_key = jax.random.split(key)
            next_frame, velocity = system.step_forward(current_frame, velocity, iter_key)

            assert not jnp.isnan(next_frame).any()

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

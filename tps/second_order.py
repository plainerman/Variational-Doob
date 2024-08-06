import jax
import jax.numpy as jnp
from tqdm import tqdm

from utils.plot import human_format

MAX_STEPS = 2_000
MAX_ABS_VALUE = 5


class SecondOrderSystem:
    def __init__(self, start_state, target_state, step_forward, step_backward, sample_velocity):
        self.start_state = start_state
        self.target_state = target_state
        self.step_forward = step_forward
        self.step_backward = step_backward
        self.sample_velocity = sample_velocity


def one_way_shooting(system, trajectory, previous_velocities, fixed_length, dt, key):
    key = jax.random.split(key)

    if previous_velocities is None:
        previous_velocities = [(trajectory[i] - trajectory[i - 1]) / dt for i in range(1, len(trajectory))]
        previous_velocities.insert(0, system.sample_velocity(key[0]))

    # pick a random point along the trajectory
    point_idx = jax.random.randint(key[1], (1,), 1, len(trajectory) - 1)[0]
    # pick a random direction, either forward or backward
    direction = jax.random.randint(key[2], (1,), 0, 2)[0]

    if direction == 0:
        trajectory = trajectory[:point_idx + 1]
        new_velocities = previous_velocities[:point_idx + 1]
        step_function = system.step_forward
    else:  # direction == 1:
        trajectory = trajectory[point_idx:][::-1]
        new_velocities = previous_velocities[point_idx:][::-1]
        step_function = system.step_backward

    steps = MAX_STEPS if fixed_length == 0 else fixed_length

    key, iter_key = jax.random.split(key[3])
    while len(trajectory) < steps:
        key, iter_key = jax.random.split(key)
        point, velocity = step_function(trajectory[-1], new_velocities[-1], iter_key)

        nan_filter = jnp.isnan(point).any(axis=-1).flatten() | jnp.isnan(velocity).any(axis=-1).flatten()
        too_big_filter = (jnp.abs(point) > MAX_ABS_VALUE).any(axis=-1).flatten()

        start_state_filter = system.start_state(point)
        target_state_filter = system.target_state(point)

        all_filters_combined = start_state_filter | target_state_filter | nan_filter | too_big_filter

        limit = jnp.argmax(all_filters_combined) + 1 if all_filters_combined.any() else len(all_filters_combined)
        trajectory.extend(point[:limit])
        new_velocities.extend(velocity[:limit])

        if (nan_filter | too_big_filter)[:limit].any():
            return False, trajectory, new_velocities

        if (start_state_filter | target_state_filter)[:limit].any():
            break

    # throw away the trajectory if it's not the right length
    if len(trajectory) > steps:
        return False, trajectory[:steps], new_velocities[:steps]

    if fixed_length != 0 and len(trajectory) != fixed_length:
        return False, trajectory, new_velocities

    if system.start_state(trajectory[0]) and system.target_state(trajectory[-1]):
        return True, trajectory, new_velocities

    if system.target_state(trajectory[0]) and system.start_state(trajectory[-1]):
        return True, trajectory[::-1], new_velocities[::-1]

    return False, trajectory, new_velocities


def two_way_shooting(system, trajectory, _previous_velocities, fixed_length, _dt, key):
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

        nan_filter = jnp.isnan(point).any(axis=-1).flatten() | jnp.isnan(velocity).any(axis=-1).flatten()
        too_big_filter = (jnp.abs(point) > MAX_ABS_VALUE).any(axis=-1).flatten()

        start_state_filter = system.start_state(point)
        target_state_filter = system.target_state(point)

        all_filters_combined = start_state_filter | target_state_filter | nan_filter | too_big_filter

        limit = jnp.argmax(all_filters_combined) + 1 if all_filters_combined.any() else len(all_filters_combined)
        new_trajectory.extend(point[:limit])
        new_velocities.extend(velocity[:limit])

        if (nan_filter | too_big_filter)[:limit].any():
            return False, new_trajectory, new_velocities

        if (start_state_filter | target_state_filter)[:limit].any():
            break

    while len(new_trajectory) < steps:
        key, iter_key = jax.random.split(key)
        point, velocity = system.step_backward(new_trajectory[0], new_velocities[0], iter_key)

        nan_filter = jnp.isnan(point).any(axis=-1).flatten() | jnp.isnan(velocity).any(axis=-1).flatten()
        too_big_filter = (jnp.abs(point) > MAX_ABS_VALUE).any(axis=-1).flatten()

        start_state_filter = system.start_state(point)
        target_state_filter = system.target_state(point)

        all_filters_combined = start_state_filter | target_state_filter | nan_filter | too_big_filter

        limit = jnp.argmax(all_filters_combined) + 1 if all_filters_combined.any() else len(all_filters_combined)
        point = point[:limit]
        velocity = velocity[:limit]
        new_trajectory[:0] = point[::-1]
        new_velocities[:0] = velocity[::-1]

        if (nan_filter | too_big_filter)[:limit].any():
            return False, new_trajectory, new_velocities

        if (start_state_filter | target_state_filter)[:limit].any():
            break

    # throw away the trajectory if it's not the right length
    if len(new_trajectory) > steps:
        return False, new_trajectory[:steps], new_velocities[:steps]

    if fixed_length != 0 and len(new_trajectory) != fixed_length:
        return False, new_trajectory, new_velocities

    if system.start_state(new_trajectory[0]) and system.target_state(new_trajectory[-1]):
        return True, new_trajectory, new_velocities

    if system.target_state(new_trajectory[0]) and system.start_state(new_trajectory[-1]):
        return True, new_trajectory[::-1], new_velocities[::-1]

    return False, new_trajectory, new_velocities


def mcmc_shooting(system, proposal, initial_trajectory, num_paths, dt, key, fixed_length=0, warmup=50, stored=None,
                  max_force_evaluations=10 ** 10):
    # pick an initial trajectory
    trajectories = [initial_trajectory]
    velocities = []
    statistics = {
        'num_force_evaluations': [],
        'num_tries': 0,
        'num_metropolis_rejected': 0,
        'warmup': warmup,
        'num_paths': num_paths,
        'max_steps': MAX_STEPS,
        'max_abs_value': MAX_ABS_VALUE,
    }
    if fixed_length > 0:
        statistics['fixed_length'] = fixed_length

    if stored is not None:
        trajectories = stored['trajectories']
        velocities = stored['velocities']
        statistics = stored['statistics']

    num_tries = 0
    num_force_evaluations = 0
    num_metropolis_rejected = 0
    total_num_force_evaluations = sum(statistics['num_force_evaluations'])
    try:
        with tqdm(total=num_paths + warmup, initial=len(trajectories) - 1,
                  desc='warming up' if warmup > 0 else '') as pbar:
            while len(trajectories) <= num_paths + warmup:
                num_tries += 1
                if len(trajectories) > warmup:
                    pbar.set_description('')

                key, traj_idx_key, ikey, accept_key = jax.random.split(key, 4)
                traj_idx = jax.random.randint(traj_idx_key, (1,), warmup + 1, len(trajectories))[0]
                # during warmup, we want an iterative scheme
                traj_idx = traj_idx if traj_idx < len(trajectories) else -1

                # trajectories and velocities are one off
                found, new_trajectory, new_velocities = proposal(system,
                                                                 trajectories[traj_idx],
                                                                 velocities[traj_idx - 1] if len(
                                                                     trajectories) > 1 else None,
                                                                 fixed_length, dt, ikey)
                num_force_evaluations += len(new_trajectory) - 1
                total_num_force_evaluations += len(new_trajectory) - 1

                pbar.set_postfix({'total_force_evaluations': human_format(total_num_force_evaluations)})

                if not found:
                    continue

                ratio = len(trajectories[-1]) / len(new_trajectory)
                # The first trajectory might have a very unreasonable length, so we skip it
                if len(trajectories) == 1 or jax.random.uniform(accept_key, shape=(1,)) < ratio:
                    # only update them in the dictionary once accepted
                    # this allows us to continue the progress
                    statistics['num_tries'] += num_tries
                    statistics['num_force_evaluations'].append(num_force_evaluations)
                    statistics['num_metropolis_rejected'] += num_metropolis_rejected
                    num_tries = 0
                    num_force_evaluations = 0
                    num_metropolis_rejected = 0

                    trajectories.append(new_trajectory)
                    velocities.append(new_velocities)
                    pbar.update(1)
                else:
                    num_metropolis_rejected += 1

                if total_num_force_evaluations > max_force_evaluations:
                    print('Max force evaluations reached, stopping early')
                    break
    except KeyboardInterrupt:
        print('SIGINT received, stopping early')
        # Fix in case we stop when adding a trajectory
        if len(trajectories) > len(velocities) + 1:
            velocities.append(new_velocities)

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

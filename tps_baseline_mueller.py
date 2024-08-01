import json
from functools import partial

import jax
import jax.numpy as jnp
import os
import matplotlib.pyplot as plt

from systems import System
from tps import first_order as tps1
import numpy as np
from utils.plot import toy_plot_energy_surface, show_or_save_fig


# minima_points = jnp.array([[-0.55828035, 1.44169],
#                            # [-0.05004308, 0.46666032],
#                            [0.62361133, 0.02804632]])
# A, B = minima_points[None, 0], minima_points[None, 2]


# We use this for the initial trajectory
def interpolate(points, steps):
    def interpolate_two_points(start, stop, steps):
        t = jnp.linspace(0, 1, steps + 1).reshape(steps + 1, 1)
        interpolated_tensors = jnp.array(start) * (1 - t) + jnp.array(stop) * t
        return interpolated_tensors

    step_size = steps // (len(points) - 1)
    remaining = steps % (len(points) - 1)

    interpolation = []
    for i in range(len(points) - 1):
        cur_step_size = step_size + (1 if i < remaining else 0)
        current = interpolate_two_points(points[i], points[i + 1], cur_step_size)
        interpolation.extend(current if i == 0 else current[1:])

    return interpolation


# plot_energy_surface = partial(toy_plot_energy_surface, U=U, states=list(zip(['A', 'B'], minima_points)),
#                               xlim=jnp.array((-1.5, 0.9)), ylim=jnp.array((-0.5, 1.7)), alpha=1.0)

if __name__ == '__main__':
    # variable or fixed length?
    variable = True
    num_paths = 1000

    save_dir = f"out/baselines/mueller"
    if variable:
        save_dir += "-variable"

    os.makedirs(save_dir, exist_ok=True)

    xi = 5
    dt = 1e-4
    T = 275e-4
    N = 0 if variable else int(T / dt)

    system = System.from_name('mueller_brown', float('inf'))
    initial_trajectory = [t.reshape(1, 2) for t in interpolate(jnp.array([system.A, system.B]), 100 if variable else N)]

    @jax.jit
    def step(_x, _key):
        """Perform one step of forward euler"""
        return _x - dt * system.dUdx(_x) + jnp.sqrt(dt) * xi * jax.random.normal(_key, _x.shape)


    tps_config = tps1.FirstOrderSystem(
        jax.jit(lambda s: jnp.linalg.norm(s - system.A) <= 0.1),
        jax.jit(lambda s: jnp.linalg.norm(s - system.B) <= 0.1),
        step
    )

    for method, name in [
        (tps1.one_way_shooting, 'one-way-shooting'),
        (tps1.two_way_shooting, 'two-way-shooting'),
    ]:
        if os.path.exists(f'{save_dir}/paths-{name}.npy') and os.path.exists(f'{save_dir}/stats-{name}.json'):
            print(f"Skipping {name} because the results are already present")

            paths = np.load(f'{save_dir}/paths-{name}.npy', allow_pickle=True)
            paths = [jnp.array(p.astype(np.float32)) for p in paths]
            with open(f'{save_dir}/stats-{name}.json', 'r') as fp:
                statistics = json.load(fp)
        else:
            paths, statistics = tps1.mcmc_shooting(tps_config, method, initial_trajectory, num_paths,
                                                   jax.random.PRNGKey(1), warmup=0, fixed_length=N)

            paths = [jnp.array(p) for p in paths]

            np.save(f'{save_dir}/paths-{name}.npy', np.array(paths, dtype=object), allow_pickle=True)
            with open(f'{save_dir}/stats-{name}.json', 'w') as fp:
                json.dump(statistics, fp)

        system.plot(trajectories=paths)
        show_or_save_fig(save_dir, f'mueller-{name}', 'pdf')

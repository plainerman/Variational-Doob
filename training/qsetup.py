import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional
from flax import linen as nn
from flax.training.train_state import TrainState
from systems import System
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax
from tqdm import trange
import utils.aldp as aldp


@dataclass
class QSetup(ABC):
    """
    This class is a container and a wrapper for all relevant information for a given setup.
    It basically wraps a specified neural network so that it follows the target output of q (mu, sigma, weights).
    Since the loss and the final layers can change, different instantiations of this class exist.
    """
    system: System
    model_q: nn.Module
    xi: ArrayLike

    @abstractmethod
    def construct_loss(self, *args, **kwargs) -> Callable:
        raise NotImplementedError

    def sample_paths(self, state_q: TrainState, x_0: ArrayLike, dt: float, T: float, BS: int,
                     key: Optional[ArrayLike], *args, **kwargs) -> ArrayLike:
        """Sample paths. If key is None, the sampling is deterministic. Otherwise, it is stochastic."""
        assert x_0.ndim == 2
        assert T / dt == int(T / dt), "dt must divide T evenly"
        N = int(T / dt)

        num_paths = x_0.shape[0]
        ndim = x_0.shape[1]
        x_t = jnp.zeros((num_paths, N, ndim), dtype=jnp.float32)
        x_t = x_t.at[:, 0, :].set(x_0)

        t = jnp.zeros((BS, 1), dtype=jnp.float32)
        u = jax.jit(lambda _t, _x: self.u_t(state_q, _t, _x, key is None, *args, **kwargs))

        for i in trange(N):
            for j in range(0, num_paths, BS):
                # If the BS does not divide the number of paths, we need to pad the last batch
                if j + BS > num_paths:
                    j_end = num_paths
                    cur_x_t = jnp.pad(x_t[j:, i], pad_width=((0, BS - (num_paths - j)), (0, 0)))

                    assert cur_x_t.shape[0] == BS
                else:
                    j_end = j + BS
                    cur_x_t = x_t[j:j_end, i]

                if key is None:
                    noise = 0
                else:
                    # For stochastic sampling we compute the noise
                    key, iter_key = jax.random.split(key)
                    noise = self.xi * jax.random.normal(iter_key, shape=(BS, ndim))

                new_x = cur_x_t + dt * u(t, cur_x_t) + jnp.sqrt(dt) * noise
                x_t = x_t.at[j:j_end, i + 1, :].set(new_x[:j_end - j])

            t += dt

        return x_t

    @abstractmethod
    def u_t(self, state_q: TrainState, t: ArrayLike, x_t: ArrayLike, deterministic: bool, *args, **kwargs) -> ArrayLike:
        raise NotImplementedError


def construct(system: System, model: Optional[nn.module], xi: float, A: ArrayLike, B: ArrayLike,
              args: argparse.Namespace) -> QSetup:
    from training.setups import diagonal, lowrank

    transform = None
    if args.internal_coordinates:
        # Initialize transform with the initial state (without second order elements)
        transform = aldp.InternalCoordinateWrapper(system.A.reshape(1, -1))

    if args.parameterization == 'diagonal':
        if args.model == 'spline':
            model = diagonal.DiagonalSpline(
                args.num_points, args.spline_mode, args.T, transform, A, B, args.num_gaussians, args.trainable_weights,
                args.base_sigma
            )
        else:
            model = diagonal.DiagonalWrapper(
                model, args.T, transform, A, B, args.num_gaussians, args.trainable_weights, args.base_sigma
            )
        return diagonal.DiagonalSetup(system, model, xi, args.ode, args.T)
    elif args.parameterization == 'low_rank':
        if args.model == 'spline':
            model = lowrank.LowRankSpline(
                args.num_points, args.spline_mode, args.T, transform, A, B, args.num_gaussians, args.trainable_weights, args.base_sigma
            )
        else:
            model = lowrank.LowRankWrapper(
                model, args.T, transform, A, B, args.num_gaussians, args.trainable_weights, args.base_sigma
            )
        return lowrank.LowRankSetup(system, model, xi, args.ode, args.T)
    else:
        raise ValueError(f"Unknown parameterization: {args.parameterization}")

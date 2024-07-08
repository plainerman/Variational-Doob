import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
from flax import linen as nn
from flax.training.train_state import TrainState

from systems import System
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax
from tqdm import trange


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

    @property
    def A(self):
        return self.system.A

    @property
    def B(self):
        return self.system.B


def construct(system: System, model: nn.module, xi: float, args: argparse.Namespace) -> Tuple[
    QSetup, ArrayLike, ArrayLike]:
    """
    Construct a QSetup object based on the given arguments.
    return: QSetup, A, B
    """
    from training.setups import diagonal

    if args.ode == 'first_order':
        order = 'first'
        A = system.A
        B = system.B
    elif args.ode == 'second_order':
        order = 'second'

        # We pad the A and B matrices with zeros to account for the velocity
        A = jnp.hstack([system.A, jnp.zeros_like(system.A)], dtype=jnp.float32)
        B = jnp.hstack([system.B, jnp.zeros_like(system.B)], dtype=jnp.float32)

        xi_velocity = jnp.ones_like(system.A) * xi
        xi_pos = jnp.zeros_like(xi_velocity) + args.xi_pos_noise

        xi = jnp.concatenate((xi_pos, xi_velocity), axis=-1, dtype=jnp.float32)
        print("Setting xi to", xi)
    else:
        raise ValueError(f"Unknown ODE: {args.ode}")

    if args.parameterization == 'diagonal':
        wrapped_module = diagonal.DiagonalWrapper(
            model, args.T, A, B, args.num_gaussians, args.trainable_weights, args.base_sigma
        )
        return diagonal.DiagonalSetup(system, wrapped_module, xi, order, args.T), A, B
    else:
        raise ValueError(f"Unknown parameterization: {args.parameterization}")

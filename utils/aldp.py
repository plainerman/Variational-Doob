from jax import nn
from jax.typing import ArrayLike
from model.utils import WrappedModule
from utils.internal import InternalCoordinateTransform


def get_aldp_transform(initial_state: ArrayLike) -> InternalCoordinateTransform:
    # This is the z-matrix for the alanine molecule we are using
    z_matrix = [
        (0, [1, 4, 6]),
        (1, [4, 6, 8]),
        (2, [1, 4, 0]),
        (3, [1, 4, 0]),
        (4, [6, 8, 14]),
        (5, [4, 6, 8]),
        (7, [6, 8, 4]),
        (9, [8, 6, 4]),
        (10, [8, 6, 4]),
        (11, [10, 8, 6]),
        (12, [10, 8, 11]),
        (13, [10, 8, 11]),
        (15, [14, 8, 16]),
        (16, [14, 8, 6]),
        (17, [16, 14, 15]),
        (18, [16, 14, 8]),
        (19, [18, 16, 14]),
        (20, [18, 16, 19]),
        (21, [18, 16, 19])
    ]
    cart_indices = [8, 6, 14]

    return InternalCoordinateTransform(66, z_indices=z_matrix, cart_indices=cart_indices, data=initial_state)


class InternalCoordinateWrapper:
    transform: InternalCoordinateTransform
    ndim: int

    def __init__(self, initial_state: ArrayLike):
        self.transform = get_aldp_transform(initial_state)
        self.ndim = initial_state.shape[1]

    def to_internal(self, x: ArrayLike) -> ArrayLike:
        return self.transform.to_internal(x)

    def __call__(self, h: ArrayLike):
        mu, sigma, w_logits = h
        BS, num_gaussians, ndim = mu.shape
        # if ndim > self.ndim, then we have second order terms
        assert ndim == self.ndim or ndim == 2 * self.ndim

        # Convert mu to cartesian coordinates
        mu = mu.at[:, :, :self.ndim].set(
            self.transform.to_cartesian(
                mu[:, :, :self.ndim]
                .reshape(BS * num_gaussians, self.ndim)
            ).reshape(BS, num_gaussians, self.ndim)
        )

        return mu, sigma, w_logits

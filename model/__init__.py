from typing import Tuple
from flax import linen as nn


class MLP(nn.Module):
    hidden_dims: Tuple[int, ...]
    activation: str
    skip_connections: bool

    @nn.compact
    def __call__(self, t):
        h = t - 0.5
        if self.activation == 'tanh':
            activation = nn.tanh
        elif self.activation == 'relu':
            activation = nn.relu
        elif self.activation == 'swish':
            activation = nn.swish
        else:
            raise NotImplementedError(f"Activation {self.activation} not implemented")

        for dim in self.hidden_dims:
            h_next = nn.Dense(dim)(h)
            h_next = activation(h_next)
            if self.skip_connections:
                h = h + h_next
            else:
                h = h_next

        return h

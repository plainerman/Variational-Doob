from typing import Tuple

from flax import linen as nn


class MLP(nn.Module):
    hidden_dims: Tuple[int, ...]
    activation: str
    resnet: bool

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

        h_add = h
        for i, dim in enumerate(self.hidden_dims):
            h = nn.Dense(dim)(h)
            h = activation(h)

            if self.resnet and i % 2 == 0 and i > 0:
                h = h + h_add
                h_add = h

        return h

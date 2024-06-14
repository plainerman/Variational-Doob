from typing import Callable, Tuple
from flax.training.train_state import TrainState
import jax
from jax.typing import ArrayLike
from tqdm import trange


def train(loss_fn: Callable, state_q: TrainState, epochs: int, key: ArrayLike) -> Tuple[TrainState, list[float]]:
    @jax.jit
    def train_step(_state_q: TrainState, _key: ArrayLike) -> (TrainState, float):
        grad_fn = jax.value_and_grad(loss_fn, argnums=0)
        loss, grads = grad_fn(_state_q.params, _key)
        _state_q = _state_q.apply_gradients(grads=grads)
        return _state_q, loss

    loss_plot = []

    for _ in trange(epochs):
        key, loc_key = jax.random.split(key)
        state_q, loss = train_step(state_q, loc_key)
        loss_plot.append(loss)

    return state_q, loss_plot

from typing import Callable, Dict, Any
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import jax
from jax.typing import ArrayLike
from orbax.checkpoint import CheckpointManager
from tqdm import trange
import jax.numpy as jnp


def train(ckpt: Any, loss_fn: Callable, epochs: int, key: ArrayLike,
          checkpoint_manager: CheckpointManager) -> Dict:
    if ckpt['model'].step >= epochs:
        return ckpt

    @jax.jit
    def train_step(_state_q: TrainState, _key: ArrayLike) -> (TrainState, float):
        grad_fn = jax.value_and_grad(loss_fn, argnums=0)
        loss, grads = grad_fn(_state_q.params, _key)
        _state_q = _state_q.apply_gradients(grads=grads)
        return _state_q, loss

    with trange(ckpt['model'].step, epochs) as pbar:
        for i in pbar:
            key, loc_key = jax.random.split(key)
            ckpt['model'], loss = train_step(ckpt['model'], loc_key)
            if loss > 1e4:
                pbar.set_postfix(log_loss=f"{jnp.log10(loss):.4f}")
            else:
                pbar.set_postfix(loss=f"{loss:.4f}")
            ckpt['losses'].append(loss.item())

            if checkpoint_manager.should_save(i + 1):
                save_args = orbax_utils.save_args_from_target(ckpt)
                checkpoint_manager.save(i + 1, ckpt, save_kwargs={'save_args': save_args})

    checkpoint_manager.wait_until_finished()

    return ckpt

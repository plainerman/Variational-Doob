from typing import Union, Dict, Any, Callable, Tuple
from flax.training.train_state import TrainState
import jax
from flax.typing import FrozenVariableDict
from jax.typing import ArrayLike


def forward_and_derivatives(state_q: TrainState, t: ArrayLike,
                            params_q: Union[FrozenVariableDict, Dict[str, Any]] = None
                            ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    params = state_q.params if params_q is None else params_q

    def _func(_t):
        _mu, _sigma, _w_logits = state_q.apply_fn(params, _t)
        return (_mu.sum(0).T, _sigma.sum(0).T), (_mu, _sigma, _w_logits)

    _jac = jax.jacrev(_func, has_aux=True)
    (_dmudt, _dsigmadt), (_mu, _sigma, _w_logits) = _jac(t)
    return _mu, _sigma, _w_logits, _dmudt.squeeze(axis=-1).T, _dsigmadt.squeeze(axis=-1).T

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from flax import linen as nn

from systems import System


@dataclass
class TrainSetup(ABC):
    system: System
    model_q: nn.Module

    @abstractmethod
    def construct_loss(self, *args, **kwargs) -> Callable:
        raise NotImplementedError

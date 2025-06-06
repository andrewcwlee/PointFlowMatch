from typing import Union, Optional
import torch
from torch.optim import Optimizer

try:
    from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
except ImportError:
    # Fallback for incompatible versions
    from enum import Enum
    
    class SchedulerType(Enum):
        CONSTANT = "constant"
        CONSTANT_WITH_WARMUP = "constant_with_warmup"
        COSINE = "cosine"
        COSINE_WITH_RESTARTS = "cosine_with_restarts"
        LINEAR = "linear"
        POLYNOMIAL = "polynomial"
    
    # Simplified scheduler mapping
    TYPE_TO_SCHEDULER_FUNCTION = {
        SchedulerType.CONSTANT: torch.optim.lr_scheduler.ConstantLR,
        SchedulerType.LINEAR: torch.optim.lr_scheduler.LinearLR,
        SchedulerType.COSINE: torch.optim.lr_scheduler.CosineAnnealingLR,
    }

def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
):
    """
    Added kwargs vs diffuser's original implementation

    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)

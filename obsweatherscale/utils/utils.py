import random
from typing import Optional, Union
import torch


class RandomStateContext:
    def __init__(self):
        self.current_state = None

    def __enter__(self):
        self.current_state = torch.random.get_rng_state()
        torch.manual_seed(torch.seed())
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.random.set_rng_state(self.current_state)


def apply_random_masking(data: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    mask_shape = (1, *data.shape[1:])
    with RandomStateContext():
        random_mask = torch.bernoulli(
            torch.ones(mask_shape)*p
        ).bool().expand_as(data)
        data[random_mask] = torch.nan
    return data


def set_active_dims(
        active_dims: Optional[list[int]] = None
    ) -> torch.Tensor | slice:
    if active_dims is None:
        return slice(None)
    return torch.tensor(active_dims, requires_grad=False)


def sample_batch_idx(
    length: int,
    batch_size: int
) -> list[int]:
    return random.sample(range(length), batch_size)


def init_device(
    gpu: Optional[list[int] | int] = None,
    use_gpu: bool = True
) -> Union[torch.nn.Module, torch.device]:
    """Initialize device based on cpu/gpu and number of gpu
    Parameters
    ----------
    gpu: list of int
        List of gpus that should be used
    use_gpu: bool
        If gpu should be used at all. If false, use cpu 

    Returns
    -------
    torch.device
    """
    if torch.cuda.is_available() and use_gpu:
        if gpu is None:
            device = torch.device("cuda")
        else:
            device = torch.device(
                f"cuda:{gpu[0]}" if isinstance(gpu, list) else f"cuda:{gpu}"
            )
    else:
        device = torch.device("cpu")

    return device

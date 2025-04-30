import random
from typing import Optional, Union
import torch
import xarray as xr


class RandomStateContext:

    def __init__(self) -> None:
        self.current_state = torch.random.get_rng_state()

    def __enter__(self) -> None:
        self.current_state = torch.random.get_rng_state()
        torch.manual_seed(torch.seed())
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.random.set_rng_state(self.current_state)


def apply_random_masking(data: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    mask_shape = (1, *data.shape[1:])
    with RandomStateContext():
        random_mask = torch.bernoulli(torch.ones(mask_shape) * p).bool().expand_as(data)
        data[random_mask] = torch.nan
    return data


def set_active_dims(active_dims: Optional[list[int]] = None) -> Union[torch.Tensor, slice]:
    if active_dims is None:
        return slice(None)
    return torch.tensor(active_dims, requires_grad=False)


def sample_batch_idx(length: int, batch_size: int) -> list[int]:
    return random.sample(range(length), batch_size)


def init_device(gpu: Optional[Union[list[int], int]] = None, use_gpu: bool = True) -> torch.device:
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
            device = torch.device(f"cuda:{gpu[0]}" if isinstance(gpu, list) else f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    return device


def wrap_tensor(pred: torch.Tensor,
                dims: tuple[str, ...],
                coords: dict[str, Union[torch.Tensor, xr.DataArray]],
                name: str = "",
                realization_name: str = "realization") -> xr.Dataset:
    # Create dimensions
    if len(pred.shape) > 3:
        dims += (realization_name, )

    # Transform back to DataArray
    pred_da = xr.DataArray(pred.detach(), coords, dims, name=name)

    return pred_da.to_dataset()

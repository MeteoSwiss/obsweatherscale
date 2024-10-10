import torch


class RandomStateContext:
    def __enter__(self):
        self.current_state = torch.random.get_rng_state()
        torch.manual_seed(torch.seed())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.random.set_rng_state(self.current_state)


def set_active_dims(
        active_dims: list[int] | None = None
    ) -> torch.Tensor | slice:
    if active_dims is None:
        return slice(None)
    return torch.tensor(active_dims, requires_grad=False)

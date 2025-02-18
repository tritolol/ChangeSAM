from typing import Iterable

from torch import nn


def set_requires_grad(parameters: Iterable[nn.Parameter], requires_grad: bool):
    """
    Sets the `requires_grad` flag for all given PyTorch parameters.

    Args:
        parameters (Iterable[nn.Parameter]): The parameters to update.
        requires_grad (bool): Whether to enable gradient computation.

    Example:
        >>> set_requires_grad(model.parameters(), False)
    """
    for param in parameters:
        param.requires_grad = requires_grad
"""The MomentumIterativeMethod attack."""

from typing import Optional

import numpy as np
import torch
from cleverhans.torch.utils import optimize_linear


def momentum_iterative_method(
    model_fn,
    x: torch.Tensor,
    eps: float = 0.3,
    eps_iter: float = 0.06,
    nb_iter: int = 10,
    norm: float = np.inf,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    y: Optional[torch.Tensor] = None,
    targeted: bool = False,
    decay_factor: float = 1.0,
    sanity_checks: bool = True,
):
    """
    PyTorch implementation of Momentum Iterative Method (MIM).

    Args:
        model_fn (callable): A callable that takes an input tensor and returns the model logits.
        x (torch.Tensor): Input tensor.
        eps (float): Maximum distortion of adversarial example compared to the original input.
        eps_iter (float): Step size for each attack iteration.
        nb_iter (int): Number of attack iterations.
        norm (Union[int, float]): Order of the norm. Possible values: `np.inf`, `1`, or `2`.
        clip_min (float): Minimum input component value.
        clip_max (float): Maximum input component value.
        y (torch.Tensor, optional): Tensor with true labels. If `targeted` is True, provide the target label.
            Otherwise, use true labels or model predictions as ground truth.
        targeted (bool): If True, create a targeted attack; otherwise, an untargeted attack.
        decay_factor (float): Decay factor for the momentum term.
        sanity_checks (bool): If True, perform sanity checks on inputs and outputs.

    Returns:
        torch.Tensor: Adversarial examples as a PyTorch tensor.
    """

    if norm == 1:
        raise NotImplementedError("This attack hasn't been tested for norm=1.")

    if norm not in [np.inf, 1, 2]:
        raise ValueError("Norm order must be np.inf, 1, or 2.")

    if eps == 0:
        return x

    x = x.clone().detach().requires_grad_(True)

    if y is None:
        # Use model predictions as ground truth
        with torch.no_grad():
            _, y = torch.max(model_fn(x), 1)

    # Initialize variables
    momentum = torch.zeros_like(x)
    adv_x = x.clone()

    for i in range(nb_iter):
        # Compute loss
        logits = model_fn(adv_x)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, y)
        if targeted:
            loss = -loss

        # Compute gradient
        grad = torch.autograd.grad(loss, adv_x, retain_graph=False, create_graph=False)[
            0
        ]

        # Normalize gradient
        red_ind = list(range(1, len(grad.shape)))  # Reduce indices except batch
        grad = grad / torch.maximum(
            torch.tensor(1e-12, device=grad.device, dtype=grad.dtype),
            torch.mean(torch.abs(grad), dim=red_ind, keepdim=True),
        )

        # Update momentum
        momentum = decay_factor * momentum + grad

        # Compute perturbation and update adversarial example
        optimal_perturbation = optimize_linear(momentum, eps_iter, norm)
        adv_x = adv_x + optimal_perturbation

        # Project perturbation to epsilon ball and clip
        eta = adv_x - x
        eta = torch.clamp(eta, -eps, eps) if norm == np.inf else eta
        adv_x = x + eta

        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)

    # Perform sanity checks if enabled
    if sanity_checks:
        if clip_min is not None:
            assert torch.all(adv_x >= clip_min)
        if clip_max is not None:
            assert torch.all(adv_x <= clip_max)

    return adv_x

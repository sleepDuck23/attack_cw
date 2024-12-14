import random
from typing import Callable, Optional

import numpy as np
import torch
from scipy.fftpack import idct


def simba(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    eps: float = 0.2,
    max_queries: int = 10000,
    targeted: bool = False,
    order: str = "random",
    freq_dims: Optional[int] = None,
    pixel_attack: bool = False,
    clip_min: Optional[float] = 0.0,
    clip_max: Optional[float] = 1.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    PyTorch implementation of the Simple Black-box Attack (SimBA), as proposed in
    https://arxiv.org/abs/1905.07121.

    Args:
        model_fn (callable): A callable that takes an input tensor and returns the model probabilities.
        x (torch.Tensor): Input tensor of shape (N, C, H, W).
        y (torch.Tensor, optional): True labels if untargeted, or target labels if targeted.
        eps (float): Step size for each attack iteration.
        max_queries (int): Maximum number of model queries allowed.
        targeted (bool): If True, perform a targeted attack; otherwise, untargeted.
        order (str): The order in which coordinates are updated ('random' or 'diagonal').
        freq_dims (int, optional): Frequency dimensions for DCT basis (used if pixel_attack is False).
        pixel_attack (bool): If True, use the pixel space for perturbations; otherwise, use DCT basis.
        clip_min (float): Minimum input component value.
        clip_max (float): Maximum input component value.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        torch.Tensor: Adversarial examples as a PyTorch tensor.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if eps == 0:
        return x

    x = x.clone().detach()
    N, C, H, W = x.shape
    device = x.device
    queries = 0

    if y is None:
        # Use model predictions as ground truth
        with torch.no_grad():
            logits = model_fn(x)
            y = logits.argmax(dim=1)

    delta = torch.zeros_like(x, device=device)

    if pixel_attack:
        # Use the pixel basis (standard basis)
        indices = np.arange(C * H * W)
    else:
        # Use DCT basis
        if freq_dims is None:
            freq_dims = H  # Assume square images and use full frequency dimensions
        indices = np.arange(freq_dims * freq_dims)

    if order == "random":
        np.random.shuffle(indices)
    elif order == "diagonal":
        # Diagonal order (low to high frequency)
        pass  # TODO (not needed for our experiment)
    else:
        raise ValueError("Order must be 'random' or 'diagonal'.")

    i = 0  # Index for basis vector

    with torch.no_grad():
        logits = model_fn(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
        current_prob = probs[torch.arange(N), y]
        queries += 1

    while queries < max_queries:
        if i >= len(indices):
            # Re-shuffle or break if all indices are used
            if order == "random":
                np.random.shuffle(indices)
                i = 0
            else:
                break  # No more directions to try

        idx = indices[i]
        i += 1

        # Generate perturbation vector q
        if pixel_attack:
            q = torch.zeros_like(x)
            c = idx // (H * W)
            hw = idx % (H * W)
            h = hw // W
            w = hw % W
            q[:, c, h, w] = 1.0
        else:
            # DCT basis vector
            u = idx // freq_dims
            v = idx % freq_dims

            # Create DCT basis vector
            basis = np.zeros((freq_dims, freq_dims))
            basis[u, v] = 1.0

            # Inverse DCT to get perturbation in image space
            q = idct(idct(basis.T, norm="ortho").T, norm="ortho")
            q = torch.from_numpy(q).float().to(device)
            q = q.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, H, W)
            q = torch.nn.functional.interpolate(
                q, size=(H, W), mode="bilinear", align_corners=False
            )
            q = q.repeat(N, C, 1, 1)

        # Normalize q to have unit norm
        q = q / torch.norm(q)

        # Try adding epsilon in positive and negative direction
        for sign in [1, -1]:
            delta_try = delta + sign * eps * q
            x_adv = x + delta_try
            if clip_min is not None or clip_max is not None:
                x_adv = torch.clamp(x_adv, clip_min, clip_max)

            logits = model_fn(x_adv)
            probs = torch.nn.functional.softmax(logits, dim=1)
            new_prob = probs[torch.arange(N), y]
            queries += 1

            # Check if attack is successful
            if targeted:
                success = new_prob > current_prob
            else:
                success = new_prob < current_prob

            if success.all():
                delta = delta_try
                current_prob = new_prob
                break  # Move to next direction

        # Check if the attack is successful
        preds = logits.argmax(dim=1)
        if targeted:
            attack_success = preds == y
        else:
            attack_success = preds != y

        if attack_success.all():
            # Attack succeeded
            break

    adv_x = x + delta
    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    return adv_x


def gaussian_perturbation_attack(
    images: torch.Tensor,
    eps: float,
    mean: float = 0.0,
    seed: int = 29,
) -> torch.Tensor:
    """
    Applies a Gaussian perturbation attack to a batch of images.

    This attack adds Gaussian noise to each pixel in the images, where the noise
    is sampled from a normal distribution with specified mean and standard deviation.
    The perturbation is controlled by the `eps` parameter, which represents the
    standard deviation of the noise.

    Args:
        images (torch.Tensor): A batch of images with shape (B, C, H, W).
        eps (float, optional): Standard deviation of the Gaussian noise. Default is 0.1.
        mean (float, optional): Mean of the Gaussian noise. Default is 0.0.
        seed (int, optional): Random seed for reproducibility. Default is 29.

    Returns:
        torch.Tensor: A batch of images with Gaussian perturbations applied.
    """
    if eps == 0:
        return images
    adv_images = images.clone()
    torch.manual_seed(seed)  # For reproducibility

    # Generate Gaussian noise
    noise = torch.randn_like(adv_images) * eps + mean

    # Apply the noise to the images
    adv_images += noise

    # Optionally, clamp the images to maintain valid pixel range
    adv_images = torch.clamp(adv_images, 0.0, 1.0)

    return adv_images


def sticker_attack(
    images: torch.Tensor,
    sticker_size: float = 0.1,
    n_channels: int = 3,
    placement: str = "center",
    seed: int = 29,
) -> torch.Tensor:
    """
    Simulates a sticker attack on a batch of images, with the option to place the sticker at the center.

    Args:
        images (torch.Tensor): A batch of images with shape (B, C, H, W).
        sticker_size (float): Fraction of the image dimension for the sticker area.
        n_channels (int): Number of channels for the sticker color.
        placement (str): Placement of the sticker ("center" or "random").
        color (list): RGB values for the sticker color.
        seed (int): Random seed for reproducibility.

    Returns:
        torch.Tensor: A batch of images with the sticker applied.
    """
    adv_images = images.clone()
    batch_size, c, h, w = images.shape
    sticker_dim = int(min(h, w) * sticker_size)  # Sticker size in pixels
    random.seed(seed)  # For reproducibility
    if n_channels == 1:
        # Grayscale sticker
        flashy_colors = [[1.0]]
    else:
        flashy_colors = [
            [1.0, 1.0, 0.0],  # Bright yellow
            [0.0, 1.0, 0.0],  # Neon green
            [1.0, 0.0, 1.0],  # Neon pink
            [0.0, 1.0, 1.0],  # Bright cyan
            [1.0, 0.5, 0.0],  # Bright orange
        ]
    for i in range(batch_size):
        color = random.choice(flashy_colors)
        if placement == "center":
            # Calculate top-left corner for a centered sticker
            top = (h - sticker_dim) // 2 + random.randint(-1, 1)
            left = (w - sticker_dim) // 2 + random.randint(-1, 1)
        elif placement == "random":
            # Random placement
            top = random.randint(0, h - sticker_dim)
            left = random.randint(0, w - sticker_dim)
        else:
            raise ValueError(f"Unsupported placement option: {placement}")

        # Apply the sticker by setting pixel values in the specified region
        adv_images[i, :, top : top + sticker_dim, left : left + sticker_dim] = (
            torch.tensor(color).view(-1, 1, 1)
        )

    return adv_images

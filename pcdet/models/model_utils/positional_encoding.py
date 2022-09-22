from multiprocessing import context
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import copy
from typing import Optional, List


class SinusoidalPositionalEncoding(torch.nn.Module):
    """
    Implements the frequency-based positional encoding described
    in [Attention is All you Need][0].
    Adds sinusoids of different frequencies to a `Tensor`. A sinusoid of a
    different frequency and phase is added to each dimension of the input `Tensor`.
    This allows the attention heads to use absolute and relative positions.
    The number of timescales is equal to hidden_dim / 2 within the range
    (min_timescale, max_timescale). For each timescale, the two sinusoidal
    signals sin(timestep / timescale) and cos(timestep / timescale) are
    generated and concatenated along the hidden_dim dimension.
    [0]: https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077
    # Parameters
    tensor : `torch.Tensor`
        a Tensor with shape (batch_size, timesteps, hidden_dim).
    min_timescale : `float`, optional (default = `1.0`)
        The smallest timescale to use.
    max_timescale : `float`, optional (default = `1.0e4`)
        The largest timescale to use.
    # Returns
    `torch.Tensor`
        The input tensor augmented with the sinusoidal frequencies.
    """  # noqa

    def __init__(self, min_timescale: float = 1.0, max_timescale: float = 1.0e4):
        super().__init__()
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def forward(self, xyz_pos: torch.Tensor, hidden_dim):
        """
        xyz_pos: (N, 3)
        """
        # TODO: Another option is to specify the expected size in init, so that we can construct
        # the positional encoding beforehand, and simply add it to the input tensor in forward.
        assert hidden_dim % 6 == 0
        num_timescales = hidden_dim // 6

        timescale_range = torch.arange(num_timescales).to(xyz_pos.device).float()

        log_timescale_increments = math.log(
            float(self.max_timescale) / float(self.min_timescale)
        ) / float(num_timescales - 1)
        inverse_timescales = self.min_timescale * torch.exp(
            timescale_range * -log_timescale_increments
        )  # (num_timescales)

        scaled_x = xyz_pos[:, 0:1] * inverse_timescales[None, :]  # (N, num_timescales)
        scaled_y = xyz_pos[:, 1:2] * inverse_timescales[None, :]  # (N, num_timescales)
        scaled_z = xyz_pos[:, 2:3] * inverse_timescales[None, :]  # (N, num_timescales)

        sinusoids = torch.cat([
            torch.sin(scaled_x), torch.cos(scaled_x),
            torch.sin(scaled_y), torch.cos(scaled_y),
            torch.sin(scaled_z), torch.cos(scaled_z),
        ], dim=-1)  # (N, hidden_dim)

        return sinusoids


def sinusoidal_positional_encoding_2d(xy_pos: torch.Tensor, hidden_dim, min_timescale: float = 1.0, max_timescale: float = 1.0e4):
    """
    Implements the frequency-based positional encoding described
    in [Attention is All you Need][0].
    Adds sinusoids of different frequencies to a `Tensor`. A sinusoid of a
    different frequency and phase is added to each dimension of the input `Tensor`.
    This allows the attention heads to use absolute and relative positions.
    The number of timescales is equal to hidden_dim / 2 within the range
    (min_timescale, max_timescale). For each timescale, the two sinusoidal
    signals sin(timestep / timescale) and cos(timestep / timescale) are
    generated and concatenated along the hidden_dim dimension.
    [0]: https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077
    # Parameters
    tensor : `torch.Tensor`
        a Tensor with shape (batch_size, timesteps, hidden_dim).
    xy_pos: (N, 2)
    min_timescale : `float`, optional (default = `1.0`)
        The smallest timescale to use.
    max_timescale : `float`, optional (default = `1.0e4`)
        The largest timescale to use.
    # Returns
    `torch.Tensor`
        The input tensor augmented with the sinusoidal frequencies.
    """
    # TODO: Another option is to specify the expected size in init, so that we can construct
    # the positional encoding beforehand, and simply add it to the input tensor in forward.
    assert hidden_dim % 4 == 0
    num_timescales = hidden_dim // 4

    timescale_range = torch.arange(num_timescales).to(xy_pos.device).float()

    log_timescale_increments = math.log(
        float(max_timescale) / float(min_timescale)
    ) / float(num_timescales - 1)
    inverse_timescales = min_timescale * torch.exp(
        timescale_range * -log_timescale_increments
    )  # (num_timescales)

    scaled_x = xy_pos[:, 0:1].float() * inverse_timescales[None, :]  # (N, num_timescales)
    scaled_y = xy_pos[:, 1:2].float() * inverse_timescales[None, :]  # (N, num_timescales)

    sinusoids = torch.cat([
        torch.sin(scaled_x), torch.cos(scaled_x),
        torch.sin(scaled_y), torch.cos(scaled_y),
    ], dim=-1)  # (N, hidden_dim)

    return sinusoids

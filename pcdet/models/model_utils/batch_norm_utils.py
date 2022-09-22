import torch
import torch.nn as nn
import time


class SparseBatchNorm1d(nn.Module):
    def __init__(self, channel, **kwargs):
        super(SparseBatchNorm1d, self).__init__()

        self.batch_norm = nn.BatchNorm1d(channel, **kwargs)

    def forward(self, input):
        """
        Args:
            input:

        Returns:

        """
        if input.dim() != 4:
            raise ValueError(
                "expected 4D input (got {}D input)".format(input.dim())
            )

        # torch.backends.cudnn.enabled = False
        input_bhwc = input.permute(0, 2, 3, 1)
        mask = torch.any(input_bhwc > 0, dim=-1)
        valid_input = input_bhwc[mask, :]
        valid_out = self.batch_norm(valid_input)
        input_bhwc[mask] = valid_out
        out = input_bhwc.permute(0, 3, 1, 2)
        # torch.backends.cudnn.enabled = True

        return out


class FastSparseBatchNorm1d(nn.Module):
    def __init__(self, channel, **kwargs):
        super(FastSparseBatchNorm1d, self).__init__()

        self.batch_norm = nn.BatchNorm1d(channel, **kwargs)

    def forward(self, input):
        """
        Args:
            input:

        Returns:

        """
        if input.dim() != 4:
            raise ValueError(
                "expected 4D input (got {}D input)".format(input.dim())
            )

        torch.backends.cudnn.enabled = False
        input_bhwc = input.permute(0, 2, 3, 1)
        mask = torch.any(input_bhwc > 0, dim=-1)
        valid_input = input_bhwc[mask, :]
        valid_out = self.batch_norm(valid_input)
        input_bhwc[mask] = valid_out
        out = input_bhwc.permute(0, 3, 1, 2)
        torch.backends.cudnn.enabled = True

        return out


def get_norm_layer(name):
    if name == 'BatchNorm2d':
        return nn.BatchNorm2d
    elif name == 'SparseBatchNorm1d':
        return SparseBatchNorm1d
    elif name == 'FastSparseBatchNorm1d':
        return FastSparseBatchNorm1d
    else:
        raise ValueError(name)

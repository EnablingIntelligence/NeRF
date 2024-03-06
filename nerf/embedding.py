import torch
from torch import nn


class NeRFEmbedding(nn.Module):
    """
    Embedding used in NeRF.
    https://arxiv.org/abs/2003.08934
    """

    def __init__(self, in_channels: int, n_freq: int):
        """
        :param in_channels: channels of the input data
        :param n_freq: number of frequency bands to add to the input data
        """
        super().__init__()
        self.out_channels = in_channels * (2 * n_freq + 1)
        self.freq_bands = torch.linspace(1, 2 ** (n_freq - 1), n_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: data
        :return: data with frequency bands (NeRF embedding)
        """
        embedding = [x]

        for freq in self.freq_bands:
            for func in [torch.sin, torch.cos]:
                embedding.append(func(x * freq))

        return torch.cat(embedding, dim=-1)

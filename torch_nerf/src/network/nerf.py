"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()

        # TODO
        self.mlp1 = nn.Sequential(nn.Linear(63, feat_dim), nn.ReLU(),
                                  nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU()),
                                  nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU()),
                                  nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU()),
                                  nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU()))
                                
        self.mlp2 = nn.Sequential(nn.Linear(319, feat_dim), nn.ReLU(),
                                 nn.Linear(feat_dim, feat_dim), nn.ReLU(),
                                 nn.Linear(feat_dim, feat_dim), nn.ReLU())
        
        self.density = nn.Sequential(nn.Linear(256, 1), nn.ReLU())

        self.fc = nn.Linear(feat_dim, feat_dim)
    
        self.mlp3 = nn.Sequential(nn.Linear(283, 128), nn.ReLU(),
                                 nn.Linear(128, 3), nn.Sigmoid())


    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """

        # TODO

        res = pos

        pos = self.mlp1(pos)
        
        pos = torch.cat((pos, res), dim=1)
        pos = self.mlp2(pos)

        sigma = self.density(pos)

        pos = self.fc(pos)
        pos = torch.cat((pos, view_dir), dim=1)
        radiance = self.mlp3(pos)

        return sigma, radiance




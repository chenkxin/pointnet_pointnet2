import torch
from torch import nn

from .base_caps2cnn import ModelSphericalCaps
from .block import BaseModel


class MSVC(BaseModel):
    def __init__(self, nclasses, bandwidths, use_residual_block=True, **kwargs):
        """

        Args:
            nclasses:
            bandwidths:
            use_residual_block:
            **kwargs:

        Example:
            model = MSVC(nclasses=10, bandwidths=[32,16,8])
            x = [torch.rand([4, 6, 64, 64]),
                 torch.rand([4, 6, 32, 32]),
                 torch.rand([4, 6, 16, 16])
                 ]
            assert model(x).shape == (4, 10)
        """
        # b: (d=6,b=32) --> (d=20,b=22) --> (n=5,d=10,b=22) -->
        #    (n=5,d=10,b=7) --> (nclasses, 10, 7)
        super(MSVC, self).__init__()
        network1 = ModelSphericalCaps(
            b_in=bandwidths[0],
            primary=[
                (20, 22),  # (d_out, b_out) S^2 conv block or residual block
                (5, 10, 22),  # (n_out_caps, d_out_caps, b_out)
            ],
            hidden=[(5, 10, 7), (nclasses, 10, 7)],
            d_in=6,
            use_residual_block=use_residual_block,
        )
        # b: (d=6,b=16) --> (d=50,b=16) --> (n=5,d=10,b=7) -->
        #    (n=5,d=10,b=7) --> (nclasses, 10, 7)
        network2 = ModelSphericalCaps(
            b_in=bandwidths[1],
            primary=[
                (50, 16),  # (d_out, b_out) S^2 conv block or residual block
                (5, 10, 7),  # (n_out_caps, d_out_caps, b_out)
            ],
            hidden=[(5, 10, 7), (nclasses, 10, 7)],
            d_in=6,
            use_residual_block=use_residual_block,
        )

        # b: (d=6,b=8) --> (d=50,b=8) --> (n=5,d=10,b=7) -->
        #    (n=5,d=10,b=7) --> (nclasses, 10, 7)
        network3 = ModelSphericalCaps(
            b_in=bandwidths[2],
            primary=[
                (50, 8),  # (d_out, b_out) S^2 conv block or residual block
                (5, 10, 7),  # (n_out_caps, d_out_caps, b_out)
            ],
            hidden=[(5, 10, 7), (nclasses, 10, 7)],
            d_in=6,
            use_residual_block=use_residual_block,
        )

        self.networks = nn.ModuleList([network1, network2, network3])

        activation = nn.ReLU
        # self.fc = nn.Sequential(
        #    nn.Linear(nclasses * 3, nclasses),
        #    nn.Softmax(dim=-1)
        # )
        # self.fc = nn.Sequential(
        #     nn.Linear(nclasses * 3, 256),
        #     activation(),
        #     nn.Linear(256, 512),
        #     activation(),
        #     nn.Linear(512, 256),
        #     activation(),
        #     nn.Linear(256, nclasses),
        #     nn.Softmax(dim=-1)
        # )
        self.fc = nn.Sequential(
            nn.Linear(nclasses * 3, 256),
            activation(),
            nn.Linear(256, nclasses),
            nn.Softmax(dim=-1),
        )
        self.class_capsule = None

    def forward(self, inputs):
        """
        Two strategies:
            1: compute capsule lengths of each network, concate them and
            use mlp

            2: use batch mlp (each capsule one mlp)
        Args:
            inputs:

        Returns:

        """

        out = []
        for n, x in zip(self.networks, inputs):
            out.append(n(x))  # [(B,n_classes), ...]
        out = torch.cat(out, dim=1)  # (B,n_classes*n_networks) where n_networks=3
        out = self.fc(out)  # (B,n_classes)
        return out


class MSVCCaps(MSVC):
    def __init__(self, *args, **kwargs):
        super(MSVCCaps, self).__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs)
        return self.class_capsule

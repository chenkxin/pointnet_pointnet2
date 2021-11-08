# -*- coding: utf-8 -*-
from .base_caps2cnn import ModelSphericalCaps


class ModelCaps(ModelSphericalCaps):
    """A caps model corresponding to baseline"""

    def __init__(
        self,
        nclasses,
        n_hidden_capsules=5,
        n_capsule_dim=10,
        bw=32,
        use_residual_block=True,
    ):
        super(ModelCaps, self).__init__(
            b_in=bw,
            nclasses=nclasses,
            primary=[
                (50, 32),  # (d_out, b_out) S^2 conv block or residual block
#                (5, 10, 22),  # (n_out_caps, d_out_caps, b_out)
                (5,10,16)
            ],
            hidden=[
 #               (5, 10, 22),
 #               (5, 10, 7),
 #               (5, 10, 7),
                (5,10,16),
                (5,10,8),
                (5,10,4),
                (nclasses, 16, 4)],
            d_in=6,
#             d_in=3,
            use_residual_block=use_residual_block,
        )

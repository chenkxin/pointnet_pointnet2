import glob
import os

import torch
import torch.nn as nn
from lib.utils import get_oldest_state

from .model_baseline import (
    ModelBaseline_3d,
    ModelBaseline_SMNIST,
    ModelBaseline_SMNIST_Deep,
)
from .model_caps import ModelCaps
from .model_resnet import ModelResNet
from .model_smnist import SMNIST
from .model_msvc import MSVC, MSVCCaps


def makeModel(
    name,
    model_dir,
    nclasses=10,
    device=None,
    is_distributed=False,
    use_residual_block=True,
    continue_training=False,
    **kwargs
):
    if name == "caps":
        model = ModelCaps(nclasses, use_residual_block=use_residual_block)
    elif name == "baseline":
        model = ModelBaseline_3d(nclasses)
    elif name == "resnet":
        model = ModelResNet(nclasses)
    elif name == "smnist":
        model = SMNIST(nclasses, use_residual_block=use_residual_block)
    elif name == "smnist_baseline":
        model = ModelBaseline_SMNIST()
    elif name == "smnist_baseline_deep":
        model = ModelBaseline_SMNIST_Deep()
    # TODO: can only use three channels
    elif name == "msvc":
        model = MSVC(
            nclasses=nclasses,
            bandwidths=[32, 16, 8],
            use_residual_block=use_residual_block,
        )
    elif name == "msvc_caps":
        model = MSVCCaps(
            nclasses=nclasses,
            bandwidths=[32, 16, 8],
            use_residual_block=use_residual_block,
        )
    else:
        raise ValueError(f"Not implemented model for {name}")

    if device:
        model = model.to(device)
        if is_distributed:
            model = nn.DataParallel(model)
    #checkpoint =torch.load("/home/chenhao/caps3d/logs/smnist_task5/best_model.ckpt")
    #model.load_state_dict(checkpoint['model'])
    # continue training feature
    LAST_EPOCH = -1
    if continue_training:
        state, LAST_EPOCH = get_oldest_state(model_dir, name)
        if state:
            model.load_state_dict(state["model"])
    return LAST_EPOCH, model
    #return model

#if __name__ == "__main__":
    #model = makeModel(name='smnist_baseline')

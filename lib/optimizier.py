import torch
from lib.utils import get_oldest_state


def makeOptimizer(name, model, lr, continue_training=False, **kwargs):
    optimizer = None
    if name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"No such name for {optimizer}")
    if continue_training:
        state, _ = get_oldest_state(kwargs["model_dir"], name)
        if state is not None:
            optimizer.load_state_dict(state["optimizer"])
    return optimizer

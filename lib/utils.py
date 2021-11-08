import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from monty.collections import AttrDict
from tqdm import tqdm

def show_recon(image, label, index, data = False):
    """
    A script for viewing spherical images of mnist
    """
    import pickle, gzip, matplotlib.pyplot as plt
    from mayavi import mlab
    import math
    def create_shpere(b=60):
        # Make sphere, choose colors
        phi, theta = np.mgrid[0 : np.pi : b * 1j, 0 : 2 * np.pi : b * 1j]
        x, y, z = np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
        return x, y, z

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(1000, 800))
    #index = 3
    x, y, z = create_shpere()
    im = np.array(image.detach().cpu().numpy(), dtype=np.dtype("uint8"))
    #mlab.mesh(x, y + n * 2 + 0.2, z, scalars=im, colormap="coolwarm")
    mlab.mesh(x, y, z, scalars=im, colormap="coolwarm")
    mlab.view(0,170,10)
    if not data:
      filename = "smnist_recon" + "_lable_" + str(label.item()) + '_' + str(index) + ".png"
    else:
      filename = "smnist_real" + "_lable_" + str(label.item()) + '_' + str(index)+ ".png"
    mlab.savefig(filename)
    #mlab.show()
    # directly on 2d
    #plt.imshow(im,cmap ='gray')
    #plt.show()


def train_step(model, optimizer, criterion, data, target, device, model_name):
    model.train()
    # for multi instance-one target input
    if isinstance(data, list):
        inputs = [i.to(device) for i in data]
        data = inputs
        #data = inputs[:,:3,:,:]
    else:
        data = data.to(device)
       #data = data[:,:3,:,:].to(device)
    if model_name == 'caps':
       prediction,y, x_recon, nclasses = model(data, target)
       loss = criterion(prediction, target, data, x_recon, nclasses)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       correct = torch.sum(torch.argmax(prediction, dim=1) == torch.argmax(y, dim=1))
    else:
      target = target.to(device)
      prediction = model (data)
      loss = criterion(prediction, target)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      correct = prediction.data.max(1)[1].eq(target.data).long().cpu().sum()

    return loss.item(), correct.item()


def test_step(model, criterion, data, target, device, batch_idx, model_name):
    model.eval()
    index = batch_idx
    if isinstance(data, list):
        inputs = [i.to(device) for i in data]
        data = inputs
        #data = inputs[:, :3, :, :]
    else:
        data = data.to(device)
        #data = data[:, :3, :, :].to(device)
#    B, overlap_num = target.shape
    if model_name == 'caps':
        prediction, y, x_recon, nclasses = model(data)
        label = torch.eye(nclasses).index_select(dim=0, index=target).to(device)
        loss = criterion(prediction, target, data, x_recon, nclasses)
        label = torch.argmax(label, dim=1)
        pre_label = torch.argmax(prediction, dim=1)
        #show_recon(data.view(data.size(2),data.size(2)), label, index, data = True)
        #show_recon(x_recon.view(data.size(2),data.size(2)), pre_label, index, data = False)
        correct = torch.sum(pre_label == label)
    else:
        target = target.to(device)
        prediction = model(data)
        loss = criterion(prediction, target)
        correct = prediction.data.max(1)[1].eq(target.data).long().cpu().sum()

    #pre = torch.topk(prediction.data, k=overlap_num)[1]
    #pre_reversed = torch.flip(pre, dims=[1])
    #result1 = pre.eq(target.data)
    #result2 = pre_reversed.eq(target.data)
    #right_bool = result1 | result2
    #correct = right_bool.long().cpu().sum()
    #return loss.item(), correct.item(), right_bool
    return loss.item(), correct.item()



def _update_acc_dict(acc_dict, target, right_bool):
    """update acc dict after every evaluating epoch"""
    if acc_dict is None:
        return
    target = target.reshape(-1)
    right_bool = right_bool.reshape(-1)
    for i, e in enumerate(target):
        if right_bool[i]:
            acc_dict[e.item()]["right"] += 1
        acc_dict[e.item()]["total"] += 1
        acc_dict[e.item()]["acc"] = round(
            acc_dict[e.item()]["right"] / acc_dict[e.item()]["total"], 2
        )
    return acc_dict


def _plot_acc(acc_for_every_cls):
    x = [acc_for_every_cls[i]["acc"] for i in range(10)]
    plt.bar(range(len(x)), x)
    plt.xticks(range(len(x)))
    plt.yticks(np.linspace(0, 1, 5, endpoint=True))
    plt.show()


def evaluate(model, criterion, testloader, device, model_name, **kwargs):
    """

    Args:
        model:
        criterion:
        testloader:
        device:
        **kwargs:

    Returns:

    """
    if "nclasses" in kwargs:
        nclases = kwargs["nclasses"]
        acc_for_every_cls = dict()
        for i in range(nclases):
            acc_for_every_cls[i] = {"right": 0, "total": 0, "acc": 0}
    else:
        acc_for_every_cls = None
    # for Test
    total_correct = 0
    total_loss = 0
    N = 0

    for batch_idx, (data, target) in enumerate(tqdm(testloader)):
        B = target.shape[0]
        #target = target.reshape(B, -1)
        #B, overlap_num = target.shape
        #l, correct, right_bool = test_step(model, criterion, data, target, device)
        l, correct= test_step(model, criterion, data, target, device,batch_idx, model_name)
        #acc_for_every_cls = _update_acc_dict(acc_for_every_cls, target, right_bool)
        total_loss += l
        total_correct += correct

        #N += B * overlap_num
        N += B

    #if "plot" in kwargs and kwargs["plot"]:
        #_plot_acc(acc_for_every_cls)
    acc = total_correct / N
    return AttrDict(loss=total_loss, acc=acc, acc_for_every_cls=acc_for_every_cls)


def get_learning_rate(epoch, learning_rate):
    limits = [100, 200]
    lrs = [1, 0.1, 0.01]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
    return lrs[-1] * learning_rate


def update_learning_rate(optimizer, epoch, learning_rate):
    lr = get_learning_rate(epoch, learning_rate)
    for p in optimizer.param_groups:
        p["learning_rate"] = lr


def check_dir(expr_name):
    model_dir = os.path.join("logs", expr_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        os.makedirs(os.path.join(model_dir, "tensorboard"))
    return model_dir


def get_oldest_model_path(model_dir, model_name, best=True):
    getter = lambda x: int(x.split("_")[-1].split(".")[0])
    model_paths = glob.glob(os.path.join(model_dir, f"{model_name}_*.ckpt"))
    if best:
        path = os.path.join(model_dir, "best_model.ckpt")
        if os.path.exists(path):
            return path
    if model_paths:
        model_paths = sorted(model_paths, key=getter)
        return model_paths[-1]
    else:
        return None


def get_oldest_state(model_dir, name):
    """waiting for optimization, interfaces are not good"""
    best_model_path = get_oldest_model_path(model_dir, name)
    if best_model_path:
        state = torch.load(best_model_path)
        return state, state["epoch"]
    else:
        return None, -1

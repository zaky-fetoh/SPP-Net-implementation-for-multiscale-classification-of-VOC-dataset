import torch.nn.functional as f
import torch.nn as nn
import torch as t
from model import *


def accuracy(output, target, cls_num=20, trshld=.5):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with t.no_grad():
        output = f.sigmoid(output) > trshld
        matchs = output.float() == target
        return matchs.sum(1) / cls_num


def save_model(net, ep_num,
               name='waight',
               outPath='./waights/'):
    file_name = outPath + name + str(ep_num) + '.pth'
    t.save(net.state_dict(),
           file_name)
    print('Model Saved', file_name)


def load_model(file_path, model=spp_net()):
    state_dict = t.load(file_path)
    model.load_state_dict(state_dict)
    print('Model loaded', file_path)


def train_(net, train_loader, criterion, opt_fn,
           device=t.device('cuda' if t.cuda.is_available() else 'cpu'),
           ):
    llis = list()
    alis = list()
    for imgs, target in train_loader:
        imgs = imgs.to(device=device)
        target = target.to(device=device)

        pred = net(imgs)

        loss = criterion(pred, target)

        opt_fn.zero_grad()
        loss.backward()
        opt_fn.step()

        llis.append(loss.item())
        alis.append(accuracy(pred,
                    target).sum().item() / target.size(0))

        print(llis[-1], alis[-1])

    return t.Tensor(llis), t.Tensor(alis)


@t.no_grad()
def validate_(net, val_loader, criterion,
              device=t.device('cuda' if t.cuda.is_available() else 'cpu'),
              ):
    llis = list()
    alis = list()
    for imgs, target in val_loader:
        imgs = imgs.to(device=device)
        target = target.to(device=device)

        pred = net(imgs)
        loss = criterion(pred, target)

        llis.append(loss.item())
        alis.append(accuracy(pred,
                    target).sum().item()/target.size(0))

        print(llis[-1], alis[-1])

    return [t.Tensor(llis), t.Tensor(alis)]


def train_and_validate(net,
                       epochs=10, scales=[150, 174, 200, 224],
                       criterion=None, opt_fn=None,
                       loader_giter=None, tr_path_file=None,
                       va_path_file=None,
                       device=t.device('cuda' if t.cuda.is_available() else 'cpu'),
                       ):
    tr_profile = dict()
    for e in range(epochs):
        tr_profile[e] = list()
        net.to(device=device)

        net.train()
        tr_ldr = loader_giter(tr_path_file, scales[e % len(scales)])
        tr_profile[e] += train_(net, tr_ldr, criterion, opt_fn,
                                device)

        save_model(net,e)

        net.eval()
        va_ldr = loader_giter(va_path_file, scales[e % len(scales)])
        tr_profile[e] += validate_(net, va_ldr, criterion, device)

    return tr_profile

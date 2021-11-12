import torch.optim as opt
import data_org as dorg
import torch.nn as nn
import model as modl
import torch as t


pos_weight = t.Tensor([17.4832, 21.3321, 14.4734, 21.9885, 15.6630, 26.8404,  9.6898, 10.6067,
        10.1007, 37.8609, 21.2528,  9.0459, 24.1224, 21.5736,  2.8671, 21.2528,
        33.4327, 22.2451, 20.9414, 19.7138]).cuda()


if __name__ == '__main__':
    dorg.download_and_extract()
    ldr_gt = dorg.getloader

    net = modl.spp_net().cuda()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt_fn = opt.Adam(net.parameters())

    tr_pro = modl.train_and_validate(net,scales=[100,115,125,],
                            criterion=loss_fn,opt_fn=opt_fn,
                            loader_giter=ldr_gt,
                            tr_path_file=dorg.TRAIN_SET_LABELS,
                            va_path_file=dorg.VAL_SET_LABELS,
                            )




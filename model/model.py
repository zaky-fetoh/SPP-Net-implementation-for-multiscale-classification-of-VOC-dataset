import torchvision.models as tvmodels
import torch.nn.functional as f
import torch.nn as nn
import torch as t
from model import *

param_count = lambda m: sum([p.numel() for p in m.parameters()])


class spp_net(nn.Module):
    def __init__(self, Spp_levels=4, out_clss=20,
                 backbone=None, fc_net=None,
                 backbone_outfeature=512):
        super(spp_net, self).__init__()
        self.feature = backbone if backbone is not None else tvmodels.squeezenet1_0(True).features
        self.classifier = fc_net if fc_net is not None else nn.Sequential(
            nn.Linear(get_fv_len(backbone_outfeature, Spp_levels), 64),
            nn.ReLU(), nn.Linear(64, out_clss),
        )
        self.spp = spp2D_layer(Spp_levels)

    def forward(self, X):
        X = self.feature(X)
        X = self.spp(X)
        return self.classifier(X)



if __name__ == '__main__':

    net = spp_net().cuda()
    tens = t.Tensor(5,3,224,224).cuda()
    out = net(tens)
    print(out.shape)

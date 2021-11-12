import torchvision.transforms as transf
import torch.utils.data as tdata
from data_org import *
import cv2 as cv
import torch


# a dataset object that read a raw images without further processing
class Raw_Voc_dataset(tdata.Dataset):
    def __init__(self, label_file_path):
        self.images_ids = get_ids(label_file_path)

    def __len__(self):
        return self.images_ids.__len__()

    def __getitem__(self, index):
        img = get_image(self.images_ids[index])
        mdt = get_cls_bb(self.images_ids[index])
        return img, mdt


class ready_voc_dataset(tdata.Dataset):
    def __init__(self, label_file_path, isotropic_scale=224,
                 transform=transf.Compose([
                     transf.ToPILImage(),
                     transf.RandomHorizontalFlip(),
                     transf.RandomVerticalFlip(),
                     transf.RandomRotation(20),
                 ])):
        self.raw_data = Raw_Voc_dataset(label_file_path)
        self.iso_scale = isotropic_scale
        self.transform = transform
        self.crop_trans = transf.Compose([
            transf.CenterCrop(self.iso_scale),
            transf.ToTensor(),
            transf.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return self.raw_data.__len__()

    def __getitem__(self, item):
        img, mdt = self.raw_data.__getitem__(item)
        img = isotropically_scale(img, self.iso_scale)
        img = self.crop_trans(self.transform(img))
        targetlabels = torch.zeros(20)
        for lbl, _, _, _, _, in mdt['object']:
            targetlabels[Encode[lbl]] = 1
        return img, targetlabels


def getloader(data_path, scale, batch_size=128,
              shuffle=True, ):
    dts = ready_voc_dataset(data_path, scale)
    dltr = tdata.DataLoader(dts, batch_size, shuffle, )
    # num_workers=4, pin_memory=True)
    return dltr

@torch.no_grad()
def get_bernoulli_class_dist(dtlr=getloader(TRAIN_SET_LABELS,
                                            5,batch_size=1000)):
    """
    presid = tensor([ 327.,  268.,  395.,  260.,  365.,  213.,  590.,  539.,  566.,  151.,
         269.,  632.,  237.,  265., 1994.,  269.,  171.,  257.,  273.,  290.])
    total_samples = 5717
    so pose_weight = total_samples / presid
    pose_weight = tensor([17.4832, 21.3321, 14.4734, 21.9885, 15.6630, 26.8404,  9.6898, 10.6067,
        10.1007, 37.8609, 21.2528,  9.0459, 24.1224, 21.5736,  2.8671, 21.2528,
        33.4327, 22.2451, 20.9414, 19.7138])

    :param dtlr:
    :return:
    """
    total_samples = 0
    cls_presid = torch.zeros(20)
    for _, lbls in dtlr:
        cls_presid += lbls.sum(0)
        total_samples += lbls.size(0)
    return cls_presid, total_samples



if __name__ == '__main__':
    pres, ts = get_bernoulli_class_dist()


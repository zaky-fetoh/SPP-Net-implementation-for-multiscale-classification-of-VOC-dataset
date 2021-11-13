# SPP-Net Multiscale Classification of Voc dataset

He, Kaiming, et al. "Spatial pyramid pooling in deep convolutional networks for visual recognition." IEEE transactions on pattern analysis and machine intelligence 37.9 
 
implementaion of SPP-net [paper](https://arxiv.org/abs/1406.4729) using PyTorch.
In this project a mutliscale and multilabel classifier is trained and evaluated Using Voc pascal dataset.

## Multiscale Training
Network is trained with N epochs, such that each epoch is performed with different scale.

## Loss function
binary cross entropy (with logits loss) is used with weight of positive propotional to the  probability of the class exitance for i.e) if the positive class ocurred k times of a dataset of size N the weight of positive loss is k/N.  


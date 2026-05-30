import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import configargparse
from cnn import ConvNet
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils import plot_loss_acc

def compute_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
import torch
from torch import nn
import numpy as np
from collections import Counter
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from tqdm import tqdm
import pandas as pd
import itertools
import os
import json
import gensim
import logging


def init_embedding(input_embedding):
    """
    Initialize embedding tensor with values from the uniform distribution.
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def clip_gradient(optimizer, grad_clip):
    """
    Clip gradients computed during backpropagation to prevent gradient explosion.
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, model, optimizer, best_acc, epochs_since_improvement, is_best, model_name, expname):
    """
    Save model checkpoint.
    """
    state = {'epoch': epoch,
             'best_acc': best_acc,
             'model': model,
             'optimizer': optimizer,
             'epochs_since_improvement': epochs_since_improvement,
             }
    filename = f'./Models/{model_name}/checkpoint_{expname}.pth.tar'
    torch.save(state, filename)
    # If checkpoint is the best so far, create a copy to avoid being overwritten by a subsequent worse checkpoint
    if is_best:
        torch.save(state, f'./Models/{model_name}/best_checkpoint_{expname}.pth.tar')


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, scale_factor):
    """
    Shrinks learning rate by a specified factor.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

import numpy as np
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.avg_reduce = 0.

    def update(self, val, n=1):
        self.val = val
        if n==-1:
            self.sum=val
            self.count=1
        else:
            self.sum += val * n
            self.count += n
        self.avg = self.sum / self.count

    def update_reduce(self, val):
        self.avg_reduce = val

    def __str__(self):
        fmtstr = '{name} {avg_reduce' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray
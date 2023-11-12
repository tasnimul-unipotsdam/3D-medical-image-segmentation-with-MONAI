import os
import gzip
import pickle

import torch
import numpy as np


def save(fn, a):
    with gzip.open(fn, 'wb', compresslevel=2) as f:
        pickle.dump(a, f, 2)


def load(fn):
    with gzip.open(fn, 'rb') as f:
        return pickle.load(f)


def mdir(path, verbose=True):
    try:
        os.makedirs(path)
    except FileExistsError:
        if verbose:
            print("Directory ", path, " already exists")


def create_dirs(modelname, experiment=None):
    base_path = os.path.join('./trained_models', modelname)
    model_path = os.path.join(base_path, experiment) if experiment else base_path
    mdir(model_path)
    return model_path


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def set_device(computer='windows'):
    if torch.cuda.is_available():
        if computer != "windows":
            device = "cuda:{}".format(get_free_gpu())
        else:
            device = "cuda:0"
    else:
        device = 'cpu'
    return torch.device(device)


def load_weights(model, weights_path):
    pretrained_dict = torch.load(weights_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)
    return model


def compute_averages(metrics, step):
    metric_list = []
    for k in metrics.keys():
        metric = torch.stack([torch.FloatTensor(metrics[k][-step:])]).mean()  # smoothing with history
        metric_list.append(metric)
    return tuple(metric_list)
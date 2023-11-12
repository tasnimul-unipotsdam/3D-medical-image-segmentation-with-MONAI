import torch
import torch.nn as nn


def weights_init_normal(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_uniform(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.uniform_(m.weight.data)
    elif isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight.data)
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_xavier_normal(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_normal_(m.weight.data, gain=1)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data, gain=1)
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_xavier_uniform(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=1)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=1)
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming_normal(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming_uniform(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
    elif isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print("init weights")
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'uniform':
        net.apply(weights_init_uniform)
    elif init_type == 'xavier_normal':
        net.apply(weights_init_xavier_normal)
    elif init_type == 'xavier_uniform':
        net.apply(weights_init_xavier_uniform)
    elif init_type == 'kaming_normal':
        net.apply(weights_init_kaiming_normal)
    elif init_type == 'kaming_uniform':
        net.apply(weights_init_kaiming_uniform)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

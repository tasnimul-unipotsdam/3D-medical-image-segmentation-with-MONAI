import os

import torch
import numpy as np
import torch.nn.functional as F

import utils

from model import Unet
from preporcess import prepare_test

device = utils.set_device()


def load_model(model_name, experiment, checkpoint):
    model = Unet(n_channels=1, n_classes=3, n_filters=32, drop=0.0)
    weights_path = os.path.join('./trained_models', model_name, experiment, '{}.pt'.format(checkpoint))
    model = utils.load_weights(model, weights_path)
    return model


@torch.no_grad()
def predict_masks(model_name='vanilla', experiment='model_1',
                  checkpoint='final'):

    model = load_model(model_name=model_name, experiment=experiment, checkpoint=checkpoint)

    out_path = os.path.join('./trained_models', model_name, experiment, "test_output")
    utils.mdir(out_path)

    model.to(device)

    image_list = []
    label_list = []
    probs_list = []
    confs_list = []
    preds_list = []
    accs_list = []

    test_data = prepare_test()

    for test_batch in test_data:
        inputs = test_batch["vol"]
        labels = test_batch["seg"]

        inputs, targets = inputs.to(device), labels.to(device)

        out = model(inputs)

        probs = F.softmax(out, dim=1)

        confs, preds = probs.max(dim=1, keepdim=False)
        gt = targets.argmax(1)
        accs = preds.eq(gt)

        image = inputs.cpu().numpy()
        label = gt.cpu().numpy()
        probs = probs.cpu().numpy()
        confs = confs.cpu().numpy()
        preds = preds.cpu().numpy()
        accs = accs.cpu().numpy()

        image_list.append(image)
        label_list.append(label)
        probs_list.append(probs)
        confs_list.append(confs)
        preds_list.append(preds)
        accs_list.append(accs)

    image = np.concatenate(image_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    confs = np.concatenate(confs_list, axis=0)
    preds = np.concatenate(preds_list, axis=0)
    accs = np.concatenate(accs_list, axis=0)

    print(image.shape, label.shape, probs.shape, confs.shape, preds.shape, accs.shape)

    name = "{}_{}".format(model_name, experiment)
    np.save(os.path.join(out_path, 'images_{}.npy'.format(name)), image)
    np.save(os.path.join(out_path, 'labels_{}.npy'.format(name)), label)
    np.save(os.path.join(out_path, 'probs_{}.npy'.format(name)), probs)
    np.save(os.path.join(out_path, 'confs_{}.npy'.format(name)), confs)
    np.save(os.path.join(out_path, 'preds_{}.npy'.format(name)), preds)
    np.save(os.path.join(out_path, 'accs_{}.npy'.format(name)), accs)
    print(np.unique(preds))

    print("evaluate model_.{}".format(name))


if __name__ == '__main__':
    print("evaluate model")
    predict_masks()


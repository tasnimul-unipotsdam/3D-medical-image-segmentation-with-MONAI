import os
import numpy as np
import random
import torch

from torchsummary import summary
from torch import optim
from tqdm import tqdm
import utils
import monai
from preporcess import prepare
from metrics import *
from model import Unet
import operations as ops
from torch import nn


torch.cuda.empty_cache()

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

OPTIMIZATION_PARAMS = {
    'base_lr': 1e-4,
    'weight_decay': 0.0,
    'epochs': 200,
    'decay_step': 10000,
    'gamma': 0.5,
    'checkpoint': 50,
    'criterion': 'CE',
    'scheduler': None,
    'optimizer': 'SGDM',
    'grad_clip': True,
    "matrix": "dice",
    "train_item": 210
}


class Trainer(object):
    def __init__(self, model, params=None, param_list=None, model_name='default', experiment=None):
        self.model = model.to(device)
        self.param_list = param_list or self.model.parameters()
        self.params = dict(OPTIMIZATION_PARAMS, **params) if params else OPTIMIZATION_PARAMS
        self.criterion, self.optimizer, self.matrix = self.set_optimization()
        self.model_path = utils.create_dirs(model_name, experiment)

    def set_optimization(self):
        if self.params['criterion'] == 'CE':
            criterion = nn.CrossEntropyLoss()

        elif self.params['criterion'] == 'dice_loss':
            criterion = monai.losses.DiceLoss(squared_pred=True, to_onehot_y=False, sigmoid=True)
        else:
            raise NotImplementedError("Please add the new loss ...")

        if self.params['optimizer'] == 'ADAM':
            optimizer = optim.Adam(self.param_list, lr=self.params['base_lr'],
                                   weight_decay=self.params['weight_decay'])

        elif self.params['optimizer'] == 'SGDM':
            optimizer = optim.SGD(self.param_list, lr=self.params['base_lr'], momentum=0.99,
                                  weight_decay=0.0)

        else:
            raise NotImplementedError("Please add the new optimizer ...")

        if self.params["matrix"] == "dice":
            matrix = dice_metric
        else:
            raise NotImplementedError("Please add the new matrix ...")

        return criterion, optimizer, matrix

    def train_step(self, inputs, targets):
        inputs, targets = inputs.to(device), targets.to(device)  # set tensors in GPU

        self.optimizer.zero_grad()  # 1

        output = self.model(inputs)

        loss = self.criterion(output, targets)

        loss.backward()  # 2

        self.optimizer.step()  # 3

        summary = {
            'Mode': 'Train',
            'Loss': loss.item(),
            "Dice": self.matrix(output, targets).item()
        }
        return summary

    @torch.no_grad()
    def val_step(self, inputs, targets):
        inputs, targets = inputs.to(device), targets.to(device)

        output = self.model(inputs)

        loss = self.criterion(output, targets)

        summary = {
            'Mode': 'Valid',
            'Loss': loss.item(),
            'Dice': self.matrix(output, targets).item(),
        }
        return summary

    @staticmethod
    def get_summaries():
        losses = dict(train_loss=[], val_loss=[])
        metrics = dict(train_dice=[], val_dice=[])
        return losses, metrics

    def get_learning_rate(self):
        if self.params['scheduler'] == 'exponential_decay':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                        step_size=self.params['decay_step'],
                                                        gamma=self.params['gamma'])
        elif self.params['scheduler'] == 'cyclical':
            scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
                                                          self.params['base_lr'],
                                                          max_lr=self.params.get('max_lr', 1.0),
                                                          step_size_up=self.params.get('step_up',
                                                                                       1),
                                                          step_size_down=self.params.get(
                                                              'step_down', 1))
        elif self.params['scheduler'] == 'WarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                             T_0=self.params.get(
                                                                                 'decay_step', 1.0),
                                                                             T_mult=1)

        else:
            scheduler = None
        return scheduler

    def start_training(self, train_data, valid_data):

        # create global summaries
        loss_summary, metrics_summary = self.get_summaries()

        scheduler = self.get_learning_rate()

        for epoch in tqdm(range(1, self.params['epochs'])):
            print('Staring epoch: {}'.format(epoch))
            print('Learning rate:',
                  scheduler.get_last_lr() if self.params['scheduler'] else self.params['base_lr'])
            step = 0
            # for batch in train_data:
            for batch in train_data:
                inputs = batch["vol"]
                labels = batch["seg"]

                train_summary = self.train_step(inputs, labels)
                # log metrics and loss
                loss_summary['train_loss'].append(train_summary['Loss'])
                metrics_summary['train_dice'].append(train_summary['Dice'])
                step += 1

            print('Validating ...')

            val_step = 0
            # for val_batch in valid_data:
            for val_batch in valid_data:
                val_inputs = val_batch["vol"]
                val_labels = val_batch["seg"]

                val_summary = self.val_step(val_inputs, val_labels)
                # log metrics and loss for validation
                loss_summary['val_loss'].append(val_summary['Loss'])
                metrics_summary['val_dice'].append(val_summary['Dice'])

                val_step += 1

            average_losses = utils.compute_averages(loss_summary, step)
            average_metrics = utils.compute_averages(metrics_summary, step)
            template = 'Epoch: {}, train loss: {}, val loss: {}, train dice: {}, val dice: {}'
            print(template.format(epoch, *average_losses, *average_metrics))

            if scheduler is not None:
                scheduler.step()

            if (epoch % self.params['checkpoint']) == 0 & (epoch != 0):
                torch.save(self.model.state_dict(),
                           os.path.join(self.model_path, 'checkpoint_{}.pt'.format(epoch)))

        return loss_summary, metrics_summary


def train_model(model_name, experiment, epochs=500, base_lr=0.1, loss_type='dice_loss',
                optimizer='ADAM', init_type='normal', schedule_type=None, grad_clip=False,
                checkpoint=50, decay_step=10000):
    seed = 86
    print(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    params = {
        'epochs': epochs,
        'base_lr': base_lr,
        'optimizer': optimizer,
        'scheduler': schedule_type,
        'criterion': loss_type,
        'grad_clip': grad_clip,
        'checkpoint': checkpoint,
        "decay_step": decay_step
    }
    train, valid = prepare()
    model = Unet(n_channels=1, n_classes=3, n_filters=32, drop=0.20)
    model.to(device)
    summary(model, (1, 32, 32, 32))
    ops.init_weights(model, init_type=init_type)
    trainer = Trainer(model,
                      params=params,
                      model_name=model_name,
                      experiment=experiment,
                      )
    loss_summary, metrics_summary = trainer.start_training(train, valid)
    torch.save(trainer.model.state_dict(), os.path.join(trainer.model_path, 'final.pt'))
    utils.save(os.path.join(trainer.model_path, 'losses.pz'), loss_summary)
    utils.save(os.path.join(trainer.model_path, 'metrics.pz'), metrics_summary)


if __name__ == '__main__':
    train_model(experiment='model_WarmRestarts',
                model_name='vanilla',
                epochs=1001,
                checkpoint=100,
                optimizer='ADAM',
                init_type='normal',
                schedule_type="WarmRestarts",
                decay_step=100)  # change experiment not model name

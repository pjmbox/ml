#!/usr/bin/env python

import os
import time
import argparse
import torch
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np


class AugmentParserBase(argparse.ArgumentParser):

    def __find_argument(self, v):
        v = v.replace('-', '_')
        for a in self._actions:
            if a.dest == v:
                return a
        raise ModuleNotFoundError()

    def set_argument(self, v, **kwargs):
        act = self.__find_argument(v)
        for k in kwargs:
            if k in act.__dict__:
                if type(kwargs[k]) != type(act.__dict__[k]):
                    raise TypeError()
                act.__dict__[k] = kwargs[k]


class PytorchBase:
    data_root = '../data'

    def __init__(self, model_name):
        self.model_name = model_name
        self.set_seed()

    def set_seed(self):
        torch.manual_seed(self.args.seed)

    def _get_property(self, v):
        _v = '__' + v
        if _v not in self.__dict__ or self.__dict__[_v] is None:
            _m = 'get_' + v
            self.__dict__[_v] = getattr(self, _m)()
        return self.__dict__[_v]

    @property
    def use_cuda(self):
        return not self.args.no_cuda and torch.cuda.is_available()

    @property
    def use_mps(self):
        return not self.args.no_mps and torch.backends.mps.is_available()

    @property
    def args(self):
        return self._get_property('args')

    @property
    def device(self):
        return self._get_property('device')

    @property
    def model(self):
        return self._get_property('model')

    @property
    def transform(self):
        return self._get_property('transform')

    # init functions
    ########################################
    @staticmethod
    def get_args_parser():
        parser = AugmentParserBase(description='PyTorch Example')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument('--no-mps', action='store_true', default=False, help='disables macOS GPU training')
        return parser

    def get_args(self):
        parser = self.get_args_parser()
        return parser.parse_args()

    def get_device(self):
        if self.use_cuda:
            return torch.device("cuda")
        elif self.use_mps:
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def get_model(self):
        raise NotImplemented()

    @staticmethod
    def get_train_criterion_args():
        return {}

    @staticmethod
    def get_test_criterion_args():
        return {}

    @staticmethod
    def get_loader(dataset, args):
        return torch.utils.data.DataLoader(dataset, **args)

    # parameter functions
    ########################################
    @staticmethod
    def get_cuda_default_args():
        return {'num_workers': 1, 'pin_memory': True, 'shuffle': True}

    @staticmethod
    def get_default_args():
        return {'batch_size': 1, 'shuffle': True, 'num_workers': 1}

    def load_model(self, model):
        if os.access(self.model_name, os.R_OK):
            model.load_state_dict(torch.load(self.model_name, map_location='cpu'))
            model.to(self.device)
        return model

    def save_model(self):
        if self.args.save_model:
            self.model.to('cpu')
            torch.save(self.model.state_dict(), self.model_name)


class PytorchExamBase(PytorchBase):

    def __init__(self, model_name):
        super().__init__(model_name)

    @property
    def exam_dataset(self):
        return self._get_property('exam_dataset')

    @property
    def exam_loader(self):
        return self._get_property('exam_loader')

    # init functions
    ########################################
    def get_args_parser(self):
        parser = super().get_args_parser()
        parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                            help='input batch size for training (default: 4)')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many exams to wait before logging training status')
        return parser

    @staticmethod
    def get_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def get_exam_dataset(self):
        return datasets.MNIST(root=self.data_root, train=True, download=True, transform=self.transform)

    def get_exam_loader(self):
        return self.get_loader(self.exam_dataset, self.get_exam_args())

    # parameter functions
    ########################################
    def get_exam_args(self):
        args = self.get_default_args()
        args['batch_size'] = self.args.batch_size
        args['shuffle'] = False
        args['num_workers'] = 2
        if self.use_cuda:
            args.update(self.get_cuda_default_args())
        return args

    # data view functions
    ########################################
    @staticmethod
    def image_show(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def do_exam(self):
        raise NotImplemented()


class PytorchTrainBase(PytorchBase):

    def __init__(self, model_name):
        super().__init__(model_name)

    @property
    def criterion(self):
        return self._get_property('criterion')

    @property
    def optimizer(self):
        return self._get_property('optimizer')

    @property
    def scheduler(self):
        return self._get_property('scheduler')

    @property
    def train_dataset(self):
        return self._get_property('train_dataset')

    @property
    def test_dataset(self):
        return self._get_property('test_dataset')

    @property
    def train_loader(self):
        return self._get_property('train_loader')

    @property
    def test_loader(self):
        return self._get_property('test_loader')

    @property
    def train_criterion_args(self):
        return self._get_property('train_criterion_args')

    @property
    def test_criterion_args(self):
        return self._get_property('test_criterion_args')

    # init functions
    ########################################
    def get_args_parser(self):
        parser = super().get_args_parser()
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=14, metavar='N',
                            help='number of epochs to train (default: 14)')
        parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
        parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                            help='Learning rate step gamma (default: 0.7)')
        parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
        return parser

    def get_optimizer(self):
        return torch.optim.Adadelta(self.model.parameters(), lr=self.args.lr)

    def get_scheduler(self):
        return StepLR(self.optimizer, step_size=1, gamma=self.args.gamma)

    @staticmethod
    def get_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    @staticmethod
    def get_criterion():
        return F.nll_loss

    def get_train_dataset(self):
        return datasets.MNIST(self.data_root, train=True, download=True, transform=self.transform)

    def get_test_dataset(self):
        return datasets.MNIST(self.data_root, train=False, download=True, transform=self.transform)

    def get_train_loader(self):
        return self.get_loader(self.train_dataset, self.get_train_args())

    def get_test_loader(self):
        return self.get_loader(self.test_dataset, self.get_test_args())

    @staticmethod
    def get_test_criterion_args():
        return {'reduction': 'sum'}

    # train and test functions
    ########################################
    def train_log(self, loader, epoch, batch_idx, data, loss):
        if batch_idx % self.args.log_interval == (self.args.log_interval - 1):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1,
                batch_idx * len(data),
                len(loader.dataset),
                100. * batch_idx / len(loader),
                loss.item()))
            return True
        return False

    def print_data_spec(self):
        data = self.train_loader.dataset.data
        print('Train, size: %d, shape: %s' % (data.size, data.shape))
        data = self.test_loader.dataset.data
        print(' Test, size: %d, shape: %s' % (data.size, data.shape))

    def train(self, epoch):
        self.model.train()
        for batch_idx, (input_data, target_data) in enumerate(self.train_loader):
            input_data, target_data = input_data.to(self.device), target_data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(input_data)
            loss = self.criterion(output, target_data, **self.train_criterion_args)
            loss.backward()
            self.optimizer.step()
            if self.train_log(self.train_loader, epoch, batch_idx, input_data, loss) and self.args.dry_run:
                break

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, **self.test_criterion_args).item()  # sum up batch loss
                predict = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += predict.eq(target.view_as(predict)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,
            correct,
            len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    # parameters preparation functions
    ########################################

    def get_train_args(self):
        args = self.get_default_args()
        args['batch_size'] = self.args.batch_size
        if self.use_cuda:
            args.update(self.get_cuda_default_args())
        return args

    def get_test_args(self):
        args = self.get_default_args()
        args['batch_size'] = self.args.test_batch_size
        args['shuffle'] = False
        args['num_workers'] = 1
        if self.use_cuda:
            args.update(self.get_cuda_default_args())
        return args

    # progress functions
    ########################################
    def do_train(self):
        self.print_data_spec()
        _ = self.scheduler
        for epoch in range(self.args.epochs):
            self.train(epoch)
            self.test()
            self.scheduler.step()
        self.save_model()

    def do_run(self):
        start = time.time()
        self.do_train()
        end = time.time()
        print('train用时:{}秒'.format(end - start))

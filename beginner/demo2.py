#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/10/18 16:32
# @Author  : jonas.pan
# @Email   : jonas.pan@signify.com
# @File    : demo2.py
# ---------------------

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from base.pytorch_base import PytorchTrainBase, PytorchExamBase


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DemoTrain(PytorchTrainBase):

    def __init__(self):
        super().__init__("cifar_net.pt")
        self.running_loss = 0.0

    def get_args_parser(self):
        parser = super().get_args_parser()
        parser.set_argument('log-interval', default=2000)
        parser.set_argument('epochs', default=2)
        parser.set_argument('batch-size', default=4)
        parser.set_argument('test-batch-size', default=16)
        parser.set_argument('lr', default=0.2)
        parser.set_argument('no-mps', default=True)
        return parser

    def get_train_dataset(self):
        return torchvision.datasets.CIFAR10(root=self.data_root, train=True, download=True, transform=self.transform)

    def get_test_dataset(self):
        return torchvision.datasets.CIFAR10(root=self.data_root, train=False, download=True, transform=self.transform)

    def get_train_args(self):
        a = super().get_train_args()
        a['num_workers'] = 4
        return a

    def get_test_args(self):
        a = super().get_train_args()
        a['num_workers'] = 4
        return a

    @staticmethod
    def get_test_criterion_args():
        return {}

    @staticmethod
    def get_transform():
        return transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    @staticmethod
    def get_criterion():
        return nn.CrossEntropyLoss()

    def get_optimizer(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def get_model(self):
        return self.load_model(Net())

    def train_log(self, loader, epoch, batch_idx, data, loss):
        self.running_loss += loss.item()
        if batch_idx % self.args.log_interval == (self.args.log_interval - 1):
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, self.running_loss / self.args.log_interval))
            self.running_loss = 0.0


class DemoExam(PytorchExamBase):

    def __init__(self):
        super().__init__("cifar_net.pt")
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def get_args_parser(self):
        parser = super().get_args_parser()
        # parser.set_argument('no-mps', default=True)
        return parser

    def get_model(self):
        return self.load_model(Net())

    def get_exam_dataset(self):
        return torchvision.datasets.CIFAR10(root=self.data_root, train=True, download=True, transform=self.transform)

    @staticmethod
    def get_transform():
        return torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def show_performance(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.exam_loader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

    def show_details(self):
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in self.exam_loader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                self.classes[i], 100 * class_correct[i] / class_total[i]))

    def show_labels(self, lbls):
        print('   Labels: ', ' '.join('%6s,' % self.classes[lbls[j]] for j in range(4)))

    def exam_images(self, imgs):
        outputs = self.model(imgs)
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%6s,' % self.classes[predicted[j]] for j in range(4)))

    def do_exam(self):
        # self.show_performance()
        # self.show_details()
        dataiter = iter(self.exam_loader)
        _, _ = dataiter.next()
        _, _ = dataiter.next()
        _, _ = dataiter.next()
        imgs, lbls = dataiter.next()
        self.show_labels(lbls)
        self.exam_images(imgs)
        self.image_show(torchvision.utils.make_grid(imgs))


if __name__ == '__main__':
    # DemoTrain().do_run()
    DemoExam().do_exam()

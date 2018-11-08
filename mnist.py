#-*- coding:utf-8 -*-
from __future__ import division 
from __future__ import absolute_import 
from __future__ import with_statement

import torch
import torch._utils
cuda = torch.cuda.is_available()
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
from torchvision.datasets import MNIST
from torchvision import transforms

from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable


########## 载入数据
mean, std = 0.1307, 0.3081

train_dataset = MNIST('./data/mnist', train=True, download=False,
                             transform=transforms.Compose([
                                 transforms.Resize(32), 
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))
test_dataset = MNIST('./data/mnist', train=False, download=False,
                            transform=transforms.Compose([
                                transforms.Resize(32), 
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                            ]))
n_classes = 10                            

# # print len(test_dataset)   #10000  :  1x28x28
# # print len(train_dataset)  #60000  :  1x28x28


# ###########################################################
# # ############################# Method 01：Softmax
# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


# # print len(test_loader)    #10000/256+1  ： 256x1x28x28   256
# # print len(train_loader)   #60000/256+1  ： 256x1x28x28   256
# # for data, label in test_loader:
    # # print data.shape, label.shape    #label为0,1,2,...,8,9
# # raw_input()


# # Set up the network and training parameters
# from model.networks import EmbeddingNet, ClassificationNet
# from model.metrics import AccumulatedAccuracyMetric
# from model.trainer import fit
# embedding_net = EmbeddingNet()
# model = ClassificationNet(embedding_net, n_classes=n_classes)
# if cuda:
    # model.cuda()
# loss_fn = torch.nn.NLLLoss()
# lr = 1e-2
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
# n_epochs = 20
# log_interval = 50                            
                            
# fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])                          
                            
# # 绘图
# from utils.utils import extract_embeddings, plot_embeddings
# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
# plot_embeddings(train_embeddings_cl, train_labels_cl, save_tag = 'train')
# val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
# plot_embeddings(val_embeddings_cl, val_labels_cl, save_tag = 'test')                            



############################################################                            
##################################### Method 02：SiameseNet 
from data.datasets import SiameseMNIST                         
siamese_train_dataset = SiameseMNIST(train_dataset) # Returns pairs of images and target same/different
siamese_test_dataset = SiameseMNIST(test_dataset)
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# print '*'*20
# print len(siamese_test_dataset)  #10000
# print len(siamese_train_dataset) #60000
# print len(siamese_test_loader)   #10000/128+1  :  [128x1x28x28, 128x1x28x28] 128 
# print len(siamese_train_loader)  #60000/128+1  :  [128x1x28x28, 128x1x28x28] 128

# for data, label, obj_label in siamese_test_loader:    # label为0或1,表示同一类或不同类, obj_label表示图像原始类别
    # print label, obj_label[0], obj_label[1] 
    # raw_input()

# for data, label, obj_label in siamese_test_loader:
    # print data[0].shape, data[1].shape, label.shape, obj_label[0].shape, obj_label[1].shape   
# raw_input()

# Set up the network and training parameters
from model.networks import EmbeddingNet, SiameseNet
from model.metrics import AccumulatedAccuracyMetric
from model.losses import ContrastiveLoss
from model.trainer import fit

from torch.optim import lr_scheduler
import torch.optim as optim

margin = 1.
embedding_net = EmbeddingNet(1)
model = SiameseNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 300

fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer,
           scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])




# 绘图
from utils.utils import extract_embeddings, plot_embeddings
# Set up data loaders
batch_size = 256
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_cl, train_labels_cl, save_tag = 'train')
val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_cl, val_labels_cl, save_tag = 'test')

# 保存和加载整个模型
torch.save(model, 'model.pkl')


# ############################################################
# ######################################## Method 03：Triplet

# # Set up data loaders
# from data.datasets import TripletMNIST

# triplet_train_dataset = TripletMNIST(train_dataset) # Returns triplets of images
# triplet_test_dataset = TripletMNIST(test_dataset)
# batch_size = 128
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


# # print '*'*20
# # print len(triplet_test_dataset)  #10000
# # print len(triplet_train_dataset) #60000
# # print len(triplet_test_loader)   #10000/128+1  :  (128L, 1L, 28L, 28L) (128L, 1L, 28L, 28L) (128L, 1L, 28L, 28L) []
# # print len(triplet_train_loader)  #60000/128+1  :  (128L, 1L, 28L, 28L) (128L, 1L, 28L, 28L) (128L, 1L, 28L, 28L) []
# # for data, label in triplet_test_loader:
    # # print data[0].shape, data[1].shape, data[2].shape, label  # label为空, 三张图片分别为Anchor、Positive、Negative
# # raw_input()


# # Set up the network and training parameters
# from model.networks import EmbeddingNet, TripletNet
# from model.losses import TripletLoss
# from model.trainer import fit

# margin = 1.
# embedding_net = EmbeddingNet(1)
# model = TripletNet(embedding_net)
# if cuda:
    # model.cuda()
# loss_fn = TripletLoss(margin)
# lr = 1e-3
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
# n_epochs = 20
# log_interval = 100

# fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)



# # 绘图
# from utils.utils import extract_embeddings, plot_embeddings
# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
# plot_embeddings(train_embeddings_cl, train_labels_cl, save_tag = 'tri_train')
# val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
# plot_embeddings(val_embeddings_cl, val_labels_cl, save_tag = 'tri_test')


# ###############################################################
# ################################### Method 04:pair selection

# from data.datasets import BalancedBatchSampler
# # We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
# train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=10, n_samples=25)
# test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=10, n_samples=25)

# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
# online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# # Set up the network and training parameters
# from model.trainer import fit
# from model.networks import EmbeddingNet
# from model.metrics import AccumulatedAccuracyMetric
# from model.losses import OnlineContrastiveLoss
# from utils.utils import AllPositivePairSelector, HardNegativePairSelector # Strategies for selecting pairs within a minibatch

# margin = 1.
# embedding_net = EmbeddingNet(1)
# model = embedding_net
# if cuda:
    # model.cuda()
# loss_fn = OnlineContrastiveLoss(margin, HardNegativePairSelector())
# lr = 1e-3
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
# n_epochs = 20
# log_interval = 50

# fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)



# ##############################################################
# ############################## Method 05:triplet selection

# from data.datasets import BalancedBatchSampler

# # We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
# train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=10, n_samples=25)
# test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=10, n_samples=25)

# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
# online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# # Set up the network and training parameters
# from model.trainer import fit
# from model.networks import EmbeddingNet
# from model.losses import OnlineTripletLoss
# # Strategies for selecting triplets within a minibatch
# from utils.utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector 
# from model.metrics import AverageNonzeroTripletsMetric, AccumulatedAccuracyMetric

# margin = 1.
# embedding_net = EmbeddingNet(1)
# model = embedding_net
# if cuda:
    # model.cuda()
# loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
# lr = 1e-3
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
# scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
# n_epochs = 20
# log_interval = 50

# fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, 
# metrics=[AverageNonzeroTripletsMetric(), AccumulatedAccuracyMetric()])



# ##############################################################
# ######################## Origin PyTorch0.3.1版本代码样例
# from __future__ import print_function
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.autograd import Variable
# import torch._utils
# try:
    # torch._utils._rebuild_tensor_v2
# except AttributeError:
    # def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        # tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        # tensor.requires_grad = requires_grad
        # tensor._backward_hooks = backward_hooks
        # return tensor
    # torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
# cuda = torch.cuda.is_available()
# # Training settings
# parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
# parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    # help='input batch size for training (default: 64)')
# parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    # help='input batch size for testing (default: 1000)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    # help='number of epochs to train (default: 10)')
# parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    # help='learning rate (default: 0.01)')
# parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    # help='SGD momentum (default: 0.5)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
                    # help='disables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
                    # help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    # help='how many batches to wait before logging training status')
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()

# torch.manual_seed(args.seed) #为CPU设置种子用于生成随机数，以使得结果是确定的
# if args.cuda:
    # torch.cuda.manual_seed(args.seed)#为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。


# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# """加载数据。组合数据集和采样器，提供数据上的单或多进程迭代器
# 参数：
# dataset：Dataset类型，从其中加载数据
# batch_size：int，可选。每个batch加载多少样本
# shuffle：bool，可选。为True时表示每个epoch都对数据进行洗牌
# sampler：Sampler，可选。从数据集中采样样本的方法。
# num_workers：int，可选。加载数据时使用多少子进程。默认值为0，表示在主进程中加载数据。
# collate_fn：callable，可选。
# pin_memory：bool，可选
# drop_last：bool，可选。True表示如果最后剩下不完全的batch,丢弃。False表示不丢弃。
# """
# train_loader = torch.utils.data.DataLoader(
    # datasets.MNIST('./mnist', train=True, download=True,
                   # transform=transforms.Compose([
                       # transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   # ])),
    # batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
    # datasets.MNIST('./mnist', train=False, transform=transforms.Compose([
                       # transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   # ])),
    # batch_size=args.batch_size, shuffle=True, **kwargs)


# class Net(nn.Module):
    # def __init__(self):
        # super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)#输入和输出通道数分别为1和10
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)#输入和输出通道数分别为10和20
        # self.conv2_drop = nn.Dropout2d()#随机选择输入的信道，将其设为0
        # self.fc1 = nn.Linear(320, 50)#输入的向量大小和输出的大小分别为320和50
        # self.fc2 = nn.Linear(50, 10)

    # def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))#conv->max_pool->relu
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))#conv->dropout->max_pool->relu
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))#fc->relu
        # x = F.dropout(x, training=self.training)#dropout
        # x = self.fc2(x)
        # return F.log_softmax(x)

# model = Net()
# if args.cuda:
    # model.cuda()#将所有的模型参数移动到GPU上

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# def train(epoch):
    # model.train()#把module设成training模式，对Dropout和BatchNorm有影响
    # for batch_idx, (data, target) in enumerate(train_loader):
        # if args.cuda:
            # data, target = data.cuda(), target.cuda()
        # '''
        # Variable类对Tensor对象进行封装，会保存该张量对应的梯度，以及对生成该张量的函数grad_fn的一个引用。
        # 如果该张量是用户创建的，grad_fn是None，称这样的Variable为叶子Variable。
        # '''
        # data, target = Variable(data), Variable(target)
        # optimizer.zero_grad()
        # output = model(data)
        # loss = F.nll_loss(output, target)#负log似然损失
        # loss.backward()
        # optimizer.step()
        # if batch_idx % args.log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                # epoch, batch_idx * len(data), len(train_loader.dataset),
                # 100. * batch_idx / len(train_loader), loss.data[0]))

# def test(epoch):
    # model.eval()#把module设置为评估模式，只对Dropout和BatchNorm模块有影响
    # test_loss = 0
    # correct = 0
    # for data, target in test_loader:
        # if args.cuda:
            # data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        # output = model(data)
        # test_loss += F.nll_loss(output, target).data[0]#Variable.data
        # pred = output.data.max(1)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data).cpu().sum()

    # test_loss = test_loss
    # test_loss /= len(test_loader) # loss function already averages over batch size
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        # test_loss, correct, len(test_loader.dataset),
        # 100. * correct / len(test_loader.dataset)))


# for epoch in range(1, args.epochs + 1):
    # train(epoch)
    # test(epoch)




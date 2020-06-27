"""Video clip order prediction."""
import os
import math
import itertools
import argparse
import time
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
from tensorboardX import SummaryWriter

from datasets.ucf101 import UCF101VCPDataset
from models.c3d import C3D
from models.r3d import R3DNet
from models.r21d import R2Plus1DNet
from models.vcopn import VCPN

from tqdm import tqdm
import ast
from utils import AverageMeter, calculate_accuracy


def order_class_index(order):
    """Return the index of the order in its full permutation.
    
    Args:
        order (tensor): e.g. [0,1,2]
    """
    classes = list(itertools.permutations(list(range(len(order)))))
    return classes.index(tuple(order.tolist()))


def adjacent_shuffle(x):
    # (C X T x H x W)
    tmp = torch.chunk(x, 4, dim=1)
    order = [0,1,2,3]
    ind1 = random.randint(0,3)
    ind2 = (ind1 + random.randint(0,2) + 1) % 4
    order[ind1], order[ind2] = order[ind2], order[ind1]
    x_new = torch.cat((tmp[order[0]], tmp[order[1]], tmp[order[2]], tmp[order[3]]),1)
    return x_new


def spatial_permutation(x):
    c, t, h, w = x.shape
    hm = h // 2
    wm = w // 2
    slices = []
    slices.append(x[:,:,:hm,:wm]) # A
    slices.append(x[:,:,:hm,wm:]) # B
    slices.append(x[:,:,hm:,:wm]) # C
    slices.append(x[:,:,hm:,wm:]) # D
    #order = [1,2,3,4]
    #while order == [1,2,3,4]:
    #    random.shuffle(order)
    order = [3,2,1,0]
    x_new = torch.cat((torch.cat((slices[order[0]], slices[order[1]]), 3), torch.cat((slices[order[2]], slices[order[3]]), 3)), 2)
    return x_new


def preprocess(inputs, targets):
    b, n, c, t, h, w = inputs.shape
    new_in = []
    # 0: origin, 1: rotation, 2: spatial permtation, 3: temporal shuffling, 4: remote clip
    for i in range(b):
        one_sample = inputs[i,:,:,:,:,:]
        one_label = targets[i]
        if one_label == 0:
            new_in.append(one_sample[0:3,:,:,:,:])
        elif one_label == 4:
            one_sample[1,:,:,:,:] = one_sample[3,:,:,:,:]
            new_in.append(one_sample[0:3,:,:,:,:])
        elif one_label == 1:
            one_sample[1,:,:,:,:] = torch.rot90(one_sample[1,:,:,:,:], random.randint(0,2) + 1, [2, 3])
            new_in.append(one_sample[0:3,:,:,:,:])
        elif one_label == 2:
            one_sample[1,:,:,:,:] = spatial_permutation(one_sample[1,:,:,:,:])
            new_in.append(one_sample[0:3,:,:,:,:])
        elif one_label == 3:
            one_sample[1,:,:,:,:] = adjacent_shuffle(one_sample[1,:,:,:,:])
            new_in.append(one_sample[0:3,:,:,:,:])
    return torch.stack(new_in)


def train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch):
    torch.set_grad_enabled(True)
    model.train()

    running_loss = 0.0
    correct = 0

    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    end_time = time.time()

    for i, data in enumerate(train_dataloader,1):
        # get inputs
        tuple_clips, targets = data
        inputs = tuple_clips.to(device)
        targets = targets.to(device)
        inputs = preprocess(inputs, targets)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward and backward
        outputs = model(inputs) # return logits here
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # compute loss and acc
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.data.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        print('Train epoch: [{0:3d}/{1:3d}][{2:4d}/{3:4d}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
          'lr {lr}'.format(
                epoch, args.epochs, i + 1, len(train_dataloader),
                batch_time=batch_time,
                loss=losses,
                acc=accuracies,
                lr=optimizer.param_groups[0]['lr']), end='\r')
        
        if i % args.pf == 0:
            step = (epoch-1)*len(train_dataloader) + i
            writer.add_scalar('train/CrossEntropyLoss', losses.val, step)
            writer.add_scalar('train/Accuracy', accuracies.val, step)
    # summary params and grads per eopch
    for name, param in model.named_parameters():
        writer.add_histogram('params/{}'.format(name), param, epoch)
        writer.add_histogram('grads/{}'.format(name), param.grad, epoch)
    
    # To avoid overlapping following logs because '\r' is used above
    print('')


def validate(args, model, criterion, device, val_dataloader, writer, epoch):
    torch.set_grad_enabled(False)
    model.eval()
    
    total_loss = 0.0
    correct = 0
    for i, data in enumerate(val_dataloader):
        # get inputs
        tuple_clips, targets = data
        inputs = tuple_clips.to(device)
        targets = targets.to(device)
        inputs = preprocess(inputs, targets)
        # forward
        outputs = model(inputs) # return logits here
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(val_dataloader)
    avg_acc = correct / len(val_dataloader.dataset)
    writer.add_scalar('val/CrossEntropyLoss', avg_loss, epoch)
    writer.add_scalar('val/Accuracy', avg_acc, epoch)
    print('[VAL] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss


def test(args, model, criterion, device, test_dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(test_dataloader, 1):
        # get inputs
        tuple_clips, targets = data
        inputs = tuple_clips.to(device)
        targets = targets.to(device)
        inputs = preprocess(inputs, targets)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(test_dataloader)
    avg_acc = correct / len(test_dataloader.dataset)
    print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Video Clip Order Prediction')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model', type=str, default='r3d', help='c3d/r3d/r21d')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--it', type=int, default=8, help='interval')
    parser.add_argument('--tl', type=int, default=3, help='tuple length')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log', type=str, default='logs', help='log directory')
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=8, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    parser.add_argument('--modality', default='rgb', type=str, help='currently support [rgb, res]')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    torch.backends.cudnn.benchmark = True
    # Force the pytorch to create context on the specific device 
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed_all(args.seed)

    ########### model ##############
    if args.model == 'c3d':
        base = C3D(with_classifier=False)
    elif args.model == 'r3d':
        base = R3DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    elif args.model == 'r21d':   
        base = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    vcpn = VCPN(base_network=base, feature_size=512, tuple_len=args.tl, modality=args.modality).to(device)

    if args.mode == 'train':  ########### Train #############
        if args.ckpt:  # resume training
            vcpn.load_state_dict(torch.load(args.ckpt))
            log_dir = os.path.dirname(args.ckpt)
        else:
            if args.desp:
                exp_name = '{}_{}_cl{}_it{}_tl{}_{}_{}'.format(args.model, args.modality, args.cl, args.it, args.tl, args.desp, time.strftime('%m%d'))
            else:
                exp_name = '{}_{}_cl{}_it{}_tl{}_{}'.format(args.model, args.modality, args.cl, args.it, args.tl, time.strftime('%m%d'))
            log_dir = os.path.join(args.log, exp_name)
            print(exp_name)
        writer = SummaryWriter(log_dir)

        train_transforms = transforms.Compose([
            transforms.Resize((128, 171)),  # smaller edge to 128
            transforms.RandomCrop(112),
            transforms.ToTensor()
        ]) 
        train_dataset = UCF101VCPDataset('data/', args.cl, args.it, args.tl, True, train_transforms)
        # split val for 800 videos
        train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset)-800, 800))
        print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                    num_workers=args.workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)

        if args.ckpt:
            pass
        else:
            # save graph and clips_order samples
            for i, data in enumerate(train_dataloader):
                tuple_clips, targets = data
                for i in range(args.tl):
                    writer.add_video('train/tuple_clips', tuple_clips[:, i, :, :, :, :], i, fps=8)
                tuple_clips = tuple_clips.to(device)
                writer.add_graph(vcpn, tuple_clips)
                break
            # save init params at step 0
            for name, param in vcpn.named_parameters():
                writer.add_histogram('params/{}'.format(name), param, 0)

        ### loss funciton, optimizer and scheduler ###
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(vcpn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=50, factor=0.1)

        prev_best_val_loss = float('inf')
        prev_best_model_path = None
        for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
            time_start = time.time()
            train(args, vcpn, criterion, optimizer, device, train_dataloader, writer, epoch)
            print('[{0:3d}/{1:3d}]Epoch time: {2:.2f} s.'.format(epoch, args.epochs, time.time() - time_start))
            val_loss = validate(args, vcpn, criterion, device, val_dataloader, writer, epoch)
            # scheduler.step(val_loss)         
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            # save model every 20 epoches
            if epoch % 20 == 0:
                torch.save(vcpn.state_dict(), os.path.join(log_dir, 'model_{}.pt'.format(epoch)))
            # save model for the best val
            if val_loss < prev_best_val_loss:
                model_path = os.path.join(log_dir, 'best_model_{}.pt'.format(epoch))
                torch.save(vcpn.state_dict(), model_path)
                prev_best_val_loss = val_loss
                if prev_best_model_path:
                    os.remove(prev_best_model_path)
                prev_best_model_path = model_path

    elif args.mode == 'test':  ########### Test #############
        vcpn.load_state_dict(torch.load(args.ckpt))
        test_transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.ToTensor()
        ])
        test_dataset = UCF101VCPDataset('data/', args.cl, args.it, args.tl, False, test_transforms)
        test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
        print('TEST video number: {}.'.format(len(test_dataset)))
        criterion = nn.CrossEntropyLoss()
        test(args, vcpn, criterion, device, test_dataloader)

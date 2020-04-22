from data import *
from utils.augmentations import SSDAugmentation
from utils import get_device
import os
import time
import random
import tools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

from models.fcos_lite import FCOS_LITE

parser = argparse.ArgumentParser(description='FCOS-LITE Detection')
parser.add_argument('-v', '--version', default='fcos_lite',
                    help='fcos_lite')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument('-hr', '--high_resolution', type=int, default=0,
                    help='1: use high resolution to pretrain; 0: else not.')  
parser.add_argument('--batch_size', default=32, type=int, 
                    help='Batch size for training')
parser.add_argument('--lr', default=1e-3, type=float, 
                    help='initial learning rate')
parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                    help='cancel warmup')
parser.add_argument('--wp_epoch', type=int, default=6,
                    help='The upper bound of warm-up')
parser.add_argument('--dataset_root', default=VOC_ROOT, 
                    help='Location of VOC root directory')
parser.add_argument('--num_classes', default=20, type=int, 
                    help='The number of dataset classes')
parser.add_argument('--momentum', default=0.9, type=float, 
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float, 
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, 
                    help='Gamma update for SGD')
parser.add_argument('--num_workers', default=8, type=int, 
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', action='store_true', default=False,  
                    help='use cuda.')
parser.add_argument('--tfboard', action='store_true', default=False,  
                    help='use tensorboard.')
parser.add_argument('--save_folder', default='weights_fcos/', type=str, 
                    help='Gamma update for SGD')

args = parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# setup_seed(20)
def train():

    cfg = voc_ab   
    hr = False   
    if args.high_resolution == 1:
        hr = True
    
    path_to_save = os.path.join(args.save_folder, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        print("use cpu")
        device = torch.device("cpu")

    # tensorboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter

        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = 'log/fcos/voc2007/' + c_time
        
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)

    # build model
    model = FCOS_LITE(device, input_size=cfg['min_dim'], num_classes=args.num_classes, trainable=True, hr=hr)
    model.to(device)
    print('Let us train FCOS-LITE on the VOC0712 dataset ......')

    # load dataset
    dataset = VOCDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # basic parameters
    input_size = cfg['min_dim']
    epoch_size = len(dataset) // args.batch_size
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay)
    
    # each part of loss weight
    cls_w = 1.0
    ctn_w = 5.0
    box_w = 1.0
 
    print("----------------------------------------Object Detection--------------------------------------------")
    print("Let's train OD network !")
    print("----------------------------------------------------------")
    print('Loading the dataset...')
    print('Training on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")


    iteration = 0
    # start training
    t0 = time.time()
    for epoch in range(cfg['max_epoch']):
        
        # step lr.
        if epoch in cfg['lr_epoch']:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)

        for iter_i, (images, targets) in enumerate(data_loader):
            iteration += 1

            # WarmUp strategy for learning rate
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = 1e-6 + (base_lr-1e-6) * (iter_i + epoch*epoch_size) / (epoch_size * (args.wp_epoch)) 
                    set_lr(optimizer, tmp_lr)
                elif epoch == args.wp_epoch:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)
            
            
            # make labels
            targets = [label.tolist() for label in targets]
            targets = tools.gt_creator(input_size=input_size, num_classes=args.num_classes, 
                                        stride=model.stride, scale_thresholds=model.scale_thresholds, 
                                        label_lists=targets)
            targets = torch.tensor(targets).float().to(device)

            # forward
            t0 = time.time()
            out = model(images.to(device))
            
            optimizer.zero_grad()
            
            cls_loss, ctn_loss, box_loss = tools.loss(out, targets, num_classes=args.num_classes)

            total_loss = cls_w * cls_loss + ctn_w * ctn_loss + box_w * box_loss
            # viz loss
            # backprop
            total_loss.backward()
            optimizer.step()

            if iter_i % 10 == 0:
                t1 = time.time()
                if args.tfboard:
                    writer.add_scalar('cls loss', cls_loss.item(), iteration)
                    writer.add_scalar('ctn loss', ctn_loss.item(), iteration)
                    writer.add_scalar('box loss', box_loss.item(), iteration)
                    writer.add_scalar('total loss', total_loss.item(), iteration)

                print('[Epoch: %d/%d][Iter: %d/%d][cls: %.4f][ctn: %.4f][box: %.4f][loss: %.4f][lr: %.6f][size: %d][time: %.6f]' 
                    % (epoch, cfg['max_epoch'], iter_i, epoch_size, cls_loss.item(), ctn_loss.item(), box_loss.item(), total_loss.item(), 
                        tmp_lr, input_size[0], t1-t0), flush=True)
                t0 = time.time()

        if (epoch + 1) % 10 == 0:
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), os.path.join(path_to_save, args.version + '_' + repr(epoch + 1) + '.pth')
                    )

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
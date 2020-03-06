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

parser = argparse.ArgumentParser(description='FCOS-LITE Detection')
parser.add_argument('-v', '--version', default='fcos_lite',
                    help='fcos_lite')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument('-hr', '--high_resolution', type=int, default=0,
                    help='1: use high resolution to pretrain; 0: else not.')  
parser.add_argument('-ms', '--multi_scale', type=int, default=0,
                    help='1: use multi-scale trick; 0: else not')                  
parser.add_argument('--batch_size', default=32, type=int, 
                    help='Batch size for training')
parser.add_argument('--lr', default=1e-3, type=float, 
                    help='initial learning rate')
parser.add_argument('-wp', '--warm_up', type=str, default='yes',
                    help='yes or no to choose using warmup strategy to train')
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
parser.add_argument('--gpu_ind', default=0, type=int, 
                    help='To choose your gpu.')
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
def train(model, device):
    global cfg, hr

    # set GPU
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if args.multi_scale == 1:
        print('Let us use the multi-scale trick.')
        ms_inds = range(len(cfg['multi_scale']))
        dataset = VOCDetection(root=args.dataset_root, transform=SSDAugmentation([608, 608], MEANS))

    else:
        dataset = VOCDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))

    from torch.utils.tensorboard import SummaryWriter
    c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    log_path = 'log/fcos/voc2007/' + c_time
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    writer = SummaryWriter(log_path)
    
    print("----------------------------------------Object Detection--------------------------------------------")
    print("Let's train OD network !")
    net = model
    net = net.to(device)

    lr = args.lr
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay)

    # loss counters
    print("----------------------------------------------------------")
    print('Loading the dataset...')
    print('Training on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    input_size = cfg['min_dim']
    step_index = 0
    epoch_size = len(dataset) // args.batch_size
    # each part of loss weight
    cls_w = 1.0
    ctn_w = 5.0
    box_w = 1.0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    iteration = 0

    # start training
    for epoch in range(cfg['max_epoch']):
        batch_iterator = iter(data_loader)
        
        # No WarmUp strategy or WarmUp stage has finished.
        if epoch in cfg['lr_epoch']:
            step_index += 1
            lr = adjust_learning_rate(optimizer, args.gamma, step_index)

        for images, targets in batch_iterator:
            # WarmUp strategy for learning rate
            if args.warm_up == 'yes':
                if epoch < args.wp_epoch:
                    lr = warmup_strategy(optimizer, epoch_size, iteration)
            iteration += 1
            
            # multi-scale trick
            if iteration % 10 == 0 and args.multi_scale == 1:
                ms_ind = random.sample(ms_inds, 1)[0]
                input_size = cfg['multi_scale'][int(ms_ind)]
            
            # multi scale
            if args.multi_scale == 1:
                images = torch.nn.functional.interpolate(images, size=input_size, mode='bilinear', align_corners=True)

            targets = [label.tolist() for label in targets]
            targets = tools.gt_creator(input_size=input_size, num_classes=args.num_classes, 
                                        stride=model.stride, scale_thresholds=model.scale_thresholds, 
                                        label_lists=targets)
            targets = torch.tensor(targets).float().to(device)

            # forward
            t0 = time.time()
            out = net(images.to(device))
            
            optimizer.zero_grad()
            
            cls_loss, ctn_loss, box_loss = tools.loss(out, targets, num_classes=args.num_classes)

            total_loss = cls_w * cls_loss + ctn_w * ctn_loss + box_w * box_loss
            # viz loss
            writer.add_scalar('cls loss', cls_loss.item(), iteration)
            writer.add_scalar('ctn loss', ctn_loss.item(), iteration)
            writer.add_scalar('box loss', box_loss.item(), iteration)
            writer.add_scalar('total loss', total_loss.item(), iteration)
            # backprop
            total_loss.backward()
            optimizer.step()
            t1 = time.time()

            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print(cls_loss.item(), ctn_loss.item(), box_loss.item())
                # print(model.s_3.item(), model.s_4.item(), model.s_5.item())
                print('Epoch[%d / %d]' % (epoch+1, cfg['max_epoch']) + ' || iter ' + repr(iteration) + \
                        ' || Loss: %.4f ||' % (total_loss.item()) + ' || lr: %.8f ||' % (lr) + ' || input size: %d ||' % input_size[0])

        if (epoch + 1) % 10 == 0:
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), args.save_folder+ '/' + args.version + '_' +
                    repr(epoch + 1) + '.pth')

def adjust_learning_rate(optimizer, gamma, step_index):
    lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def warmup_strategy(optimizer, epoch_size, iteration):
    lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * (args.wp_epoch)) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    global hr, cfg

    hr = False
    device = get_device(args.gpu_ind)
    
    if args.high_resolution == 1:
        hr = True
    
    cfg = voc_ab

    if args.version == 'fcos_lite':
        from models.fcos_lite import FCOS_LITE
    
        fcos_lite = FCOS_LITE(device, input_size=cfg['min_dim'], num_classes=args.num_classes, trainable=True, hr=hr)
        print('Let us train FCOS-LITE on the VOC0712 dataset ......')

    
    train(fcos_lite, device)
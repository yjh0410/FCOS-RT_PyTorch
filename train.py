from __future__ import division

import os
import random
import argparse
import time
import math
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import *
import tools

from data import VOC_CLASSES, VOC_ROOT, VOCDetection
from data import coco_root, COCODataset
from data import config
from data import BaseTransform, detection_collate

import tools

from utils import distributed_utils
from utils.augmentations import WeakAugmentation, StrongAugmentation, ColorAugmentation
from utils.coco_evaluator import COCOAPIEvaluator
from utils.voc_evaluator import VOCAPIEvaluator
from utils.modules import ModelEMA


def parse_args():
    parser = argparse.ArgumentParser(description='FCOS Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size for training')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--num_gpu', default=1, type=int, 
                        help='Number of GPUs.')
    parser.add_argument('--wp_iters', type=int,
                            default=500, help='wram-up epoch')
    parser.add_argument('--eval_epoch', type=int,
                            default=2, help='interval between evaluations')
    parser.add_argument('--schedule', type=int,
                            default=1, help='lr schedule.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')

    # model
    parser.add_argument('-v', '--version', default='fcos',
                        help='fcos')

    # dataset
    parser.add_argument('-d', '--dataset', default='coco',
                        help='voc or coco')

    # train trick
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')      
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='use mosaic augmentation')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema training trick')

    # train DDP
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--local_rank', type=int, default=0, 
                        help='local_rank')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')


    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    # config file
    if args.version == 'fcos':
        cfg = config.train_cfg
        
    # model name
    model_name = args.version
    print('Model: ', model_name)

    # set distributed
    local_rank = 0
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = torch.distributed.get_rank()
        print(local_rank)
        torch.cuda.set_device(local_rank)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)
    
    # mosaic augmentation
    if args.mosaic:
        print('use Mosaic Augmentation ...')

    # multi-scale
    if args.multi_scale:
        print('use the multi-scale trick ...')
        train_size = cfg['img_size']
        val_size = cfg['val_size']
    else:
        train_size = cfg['train_size']
        val_size = cfg['val_size']

    # EMA trick
    if args.ema:
        print('use EMA trick ...')

    # build augmentation
    if args.schedule < 1:
        print('Unknow schedule !!')
        exit(0)
    elif args.schedule == 1:
        # 1x schedule
        augmentation = WeakAugmentation(train_size)
    else:
        # train more than 1x
        augmentation = StrongAugmentation(train_size)

    # dataset and evaluator
    if args.dataset == 'voc':
        data_dir = VOC_ROOT
        num_classes = 20
        dataset = VOCDetection(root=data_dir, 
                                img_size=train_size,
                                transform=augmentation,
                                base_transform=ColorAugmentation(train_size),
                                mosaic=args.mosaic
                                )

        evaluator = VOCAPIEvaluator(data_root=data_dir,
                                    img_size=val_size,
                                    device=device,
                                    transform=BaseTransform(val_size),
                                    labelmap=VOC_CLASSES
                                    )

    elif args.dataset == 'coco':
        data_dir = coco_root
        num_classes = 80
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=train_size,
                    transform=augmentation,
                    base_transform=ColorAugmentation(train_size),
                    mosaic=args.mosaic)

        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=BaseTransform(val_size)
                        )
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)
    
    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # buile model and config file
    if model_name == 'fcos':
        from models.fcos import FCOS
        backbone = cfg['backbone']
        # model
        net = FCOS(device=device, 
                    img_size=train_size, 
                    num_classes=num_classes, 
                    trainable=True, 
                    bk=backbone
                    )
    
    else:
        print('Unknown model name...')
        exit(0)

    model = net

    # SyncBatchNorm
    if args.sybn and args.cuda and args.num_gpu > 1:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(device)


    # distributed
    if args.distributed and args.num_gpu > 1:
        print('using DDP ...')
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        batch_size=args.batch_size, 
                        collate_fn=detection_collate,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        sampler=torch.utils.data.distributed.DistributedSampler(dataset)
                        )

    else:
        # train on 1 GPU
        model = model.train().to(device)
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset,
                        shuffle=True,
                        batch_size=args.batch_size, 
                        collate_fn=detection_collate,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )

    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        if args.distributed:
            model.module.load_state_dict(torch.load(args.resume, map_location=device))
        else:
            model.load_state_dict(torch.load(args.resume, map_location=device))

    # EMA
    ema = ModelEMA(model) if args.ema else None

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, c_time)
        os.makedirs(log_path, exist_ok=True)

        tblogger = SummaryWriter(log_path)
    
    batch_size = args.batch_size
    max_epoch = cfg['max_epoch'] * args.schedule
    lr_epoch = [e * args.schedule for e in cfg['lr_epoch']]
    epoch_size = len(dataset) // (batch_size * args.num_gpu)
    print('Max epoch: ', max_epoch)
    print('Lr epoch:', lr_epoch)

    # build optimizer
    base_lr = cfg['lr']
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), 
                            lr=tmp_lr, 
                            momentum=0.9,
                            weight_decay=1e-4
                            )
    
    best_map = 0.
    t0 = time.time()
    warmup = True

    # start to train
    for epoch in range(args.start_epoch, max_epoch):
        # set epoch if DDP
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)

        # use step lr
        if epoch in lr_epoch:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)

        # load a batch
        for iter_i, (images, targets) in enumerate(dataloader):
            # WarmUp strategy for learning rate
            ni = iter_i + epoch_size*epoch
            if ni < args.wp_iters and warmup:
                tmp_lr = base_lr * pow((ni) / (args.wp_iters), 4)
                set_lr(optimizer, tmp_lr)

            elif ni >= args.wp_iters and warmup:
                # warmup is over
                warmup = False
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                r = cfg['random_size_range']
                train_size = random.randint(r[0], r[1]) * 128
                model.module.set_grid(train_size) if args.distributed else model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            
            # make labels
            targets = [label.tolist() for label in targets]
            # vis_data(images, targets, train_size)
            # continue
            targets = tools.gt_creator(img_size=train_size,
                                       num_classes=num_classes, 
                                       strides=net.strides, 
                                       scale_range=cfg['scale_range'],
                                       label_lists=targets
                                       )
                                        
            # to device
            images = images.to(device)
            targets = targets.to(device)

            # forward
            cls_loss, reg_loss, ctn_loss, total_loss = model(images, targets=targets)

            loss_dict = dict(
                cls_loss=cls_loss,
                reg_loss=reg_loss,
                ctn_loss=ctn_loss,
                total_loss=total_loss
            )
            loss_dict_reduced = distributed_utils.reduce_loss_dict(loss_dict)

            # check NAN
            if torch.isnan(total_loss):
                continue

            # backprop
            total_loss.backward()        
            optimizer.step()
            optimizer.zero_grad()

            # ema
            if args.ema:
                ema.update(model)

            # display
            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    tblogger.add_scalar('cls loss',  loss_dict_reduced['cls_loss'].item(),  iter_i + epoch * epoch_size)
                    tblogger.add_scalar('reg loss',  loss_dict_reduced['reg_loss'].item(),  iter_i + epoch * epoch_size)
                    tblogger.add_scalar('ctn loss',  loss_dict_reduced['ctn_loss'].item(),  iter_i + epoch * epoch_size)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f][Loss: cls %.2f || reg %.2f || ctn %.2f || size %d || time: %.2f]'
                        % (epoch+1, 
                           max_epoch, 
                           iter_i, 
                           epoch_size, 
                           tmp_lr,
                           loss_dict_reduced['cls_loss'].item(), 
                           loss_dict_reduced['reg_loss'].item(), 
                           loss_dict_reduced['ctn_loss'].item(), 
                           train_size, 
                           t1-t0),
                        flush=True)

                t0 = time.time()

        # evaluation
        if (epoch + 1) % args.eval_epoch == 0:
            if args.ema:
                model_eval = ema.ema
            else:
                model_eval = model.module if args.distributed else model

            # set eval mode
            model_eval.trainable = False
            model_eval.set_grid(val_size)
            model_eval.eval()

            if local_rank == 0:
                # evaluate
                evaluator.evaluate(model_eval)

                cur_map = evaluator.map if args.dataset == 'voc' else evaluator.ap50_95
                if cur_map > best_map:
                    # update best-map
                    best_map = cur_map
                    # save model
                    print('Saving state, epoch:', epoch + 1)
                    torch.save(model_eval.state_dict(), os.path.join(path_to_save, 
                                args.version + '_' + repr(epoch + 1) + '_' + str(round(best_map, 2)) + '.pth')
                                )  
                if args.tfboard:
                    if args.dataset == 'voc':
                        tblogger.add_scalar('07test/mAP', evaluator.map, epoch)
                    elif args.dataset == 'coco':
                        tblogger.add_scalar('val/AP50_95', evaluator.ap50_95, epoch)
                        tblogger.add_scalar('val/AP50', evaluator.ap50, epoch)

            if args.distributed:
                # wait for all processes to synchronize
                dist.barrier()

            # set train mode.
            model_eval.trainable = True
            model_eval.set_grid(train_size)
            model_eval.train()
    
    if args.tfboard:
        tblogger.close()


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def vis_data(images, targets, input_size):
    # vis data
    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    img = images[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    img = ((img * std + mean)*255).astype(np.uint8)
    cv2.imwrite('1.jpg', img)

    img_ = cv2.imread('1.jpg')
    for box in targets[0]:
        xmin, ymin, xmax, ymax = box[:-1]
        # print(xmin, ymin, xmax, ymax)
        xmin *= input_size
        ymin *= input_size
        xmax *= input_size
        ymax *= input_size
        cv2.rectangle(img_, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

    cv2.imshow('img', img_)
    cv2.waitKey(0)


if __name__ == '__main__':
    train()

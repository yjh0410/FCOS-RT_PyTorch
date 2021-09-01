from __future__ import division

import os
import random
import argparse
import time
import numpy as np
import cv2

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import VOC_CLASSES, VOC_ROOT, VOCDetection
from data import coco_root, COCODataset
from data import config
from data import BaseTransform, detection_collate

from create_gt import gt_creator

from utils import distributed_utils
from utils.augmentations import Augmentation
from utils.coco_evaluator import COCOAPIEvaluator
from utils.voc_evaluator import VOCAPIEvaluator
from utils.modules import ModelEMA
from utils.com_flops_params import FLOPs_and_Params


def parse_args():
    parser = argparse.ArgumentParser(description='FCOS-RT Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size for training')
    parser.add_argument('--start_iter', type=int, default=0,
                        help='start iteration to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--num_gpu', default=1, type=int, 
                        help='Number of GPUs.')
    parser.add_argument('--start_epoch', type=int,
                            default=0, help='the start epoch to train')
    parser.add_argument('--wp_epoch', type=int,
                            default=1, help='wram-up epoch')
    parser.add_argument('--eval_epoch', type=int,
                            default=2, help='interval between evaluations')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='visualize target.')

    # model
    parser.add_argument('-v', '--version', default='fcos_rt',
                        help='fcos_rt, fcos')
    parser.add_argument('-bk', '--backbone', default='r18',
                        help='r18, r50, r101, dla34, d53')

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
    if args.version == 'fcos_rt':
        cfg = config.fcos_rt_train_cfg
    elif args.version == 'fcos':
        cfg = config.fcos_train_cfg
        
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

    # dataset and evaluator
    if args.dataset == 'voc':
        data_dir = VOC_ROOT
        num_classes = 20
        dataset = VOCDetection(root=data_dir, 
                                img_size=train_size,
                                transform=Augmentation(train_size),
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
                    transform=Augmentation(train_size),
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
    if model_name == 'fcos_rt':
        from models.fcos_rt import FCOS_RT
        backbone = args.backbone
        # model
        net = FCOS_RT(device=device, 
                     img_size=train_size, 
                     num_classes=num_classes, 
                     trainable=True, 
                     bk=backbone,
                     freeze_bn=cfg['freeze_bn']
                     )
    
    elif model_name == 'fcos':
        from models.fcos import FCOS
        backbone = args.backbone
        # model
        net = FCOS(device=device, 
                    img_size=train_size, 
                    num_classes=num_classes, 
                    trainable=True, 
                    bk=backbone,
                    freeze_bn=cfg['freeze_bn']
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

    # compute FLOPs and Params
    FLOPs_and_Params(model=model, size=train_size)


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

    batch_iter = iter(dataloader)

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
    tblogger = None
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, c_time)
        os.makedirs(log_path, exist_ok=True)

        tblogger = SummaryWriter(log_path)
    
    # basic
    batch_size = args.batch_size
    warmup = True
    max_epoch = cfg['max_epoch']
    lr_step = cfg['lr_step']
    epoch_size = len(dataset) // (batch_size * args.num_gpu)
    print('Max epoch: ', max_epoch)
    print('Lr step:', lr_step)

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
    epoch = 0
    # start to train
    for epoch in range(args.start_epoch, max_epoch):
        # set epoch if DDP
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)

        # use step lr
        if epoch in lr_step:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)

        # load a batch
        for iter_i, (images, targets) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            # warmup
            if epoch < args.wp_epoch and warmup:
                nw = args.wp_epoch * epoch_size
                tmp_lr = base_lr * pow(ni / nw, 4)
                set_lr(optimizer, tmp_lr)

            elif epoch == args.wp_epoch and iter_i == 0 and warmup:
                # warmup is over
                warmup = False
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                r = cfg['random_size_range']
                train_size = random.randint(r[0], r[1]) * 32
                model.module.set_grid(train_size) if args.distributed else model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            
            targets = [label.tolist() for label in targets]
            # visualize target
            if args.vis:
                vis_data(images, targets, train_size)
                continue
            
            # make labels
            targets = gt_creator(img_size=train_size,
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
                    tblogger.add_scalar('cls loss',  loss_dict_reduced['cls_loss'].item(),  iter_i)
                    tblogger.add_scalar('reg loss',  loss_dict_reduced['reg_loss'].item(),  iter_i)
                    tblogger.add_scalar('ctn loss',  loss_dict_reduced['ctn_loss'].item(),  iter_i)
                
                t1 = time.time()
                print('[Epoch %d][Iter %d/%d][lr %.6f][Loss: cls %.2f || reg %.2f || ctn %.2f || size %d || time: %.2f]'
                        % (epoch, 
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
            # update iter_i
            iter_i += 1
    
        # evaluate
        if args.ema:
            model_eval = ema.ema
        else:
            model_eval = model.module if args.distributed else model

        best_map = eval(model=model_eval,
                        train_size=train_size,
                        val_size=val_size,
                        path_to_save=path_to_save,
                        epoch=epoch,
                        best_map=best_map,
                        evaluator=evaluator,
                        tblogger=tblogger,
                        local_rank=local_rank,
                        ddp=args.distributed,
                        dataset=args.dataset,
                        model_name=args.version)
        
        # rebuild batch iter
        epoch += 1

    # final evaluate
    if args.ema:
        model_eval = ema.ema
    else:
        model_eval = model.module if args.distributed else model

    best_map = eval(model=model_eval,
                    train_size=train_size,
                    val_size=val_size,
                    path_to_save=path_to_save,
                    epoch=epoch,
                    best_map=best_map,
                    evaluator=evaluator,
                    tblogger=tblogger,
                    local_rank=local_rank,
                    ddp=args.distributed,
                    dataset=args.dataset,
                    model_name=args.version)

    if args.tfboard:
        tblogger.close()


def eval(model, 
         train_size,
         val_size, 
         path_to_save, 
         epoch, 
         best_map=-1., 
         evaluator=None, 
         tblogger=None, 
         local_rank=0, 
         ddp=False,
         dataset='voc', 
         model_name='fcos'):
    # set eval mode
    model.trainable = False
    model.set_grid(val_size)
    model.eval()

    if local_rank == 0:
        if evaluator is None:
            print('continue training ...')
        else:
            # evaluate
            evaluator.evaluate(model)

            cur_map = evaluator.map if dataset == 'voc' else evaluator.ap50_95
            if cur_map > best_map:
                # update best-map
                best_map = cur_map
                # save model
                print('Saving state, epoch:', epoch + 1)
                save_name = os.path.join(path_to_save, model_name + '_' + repr(epoch + 1) + '_' + str(round(best_map, 2)) + '.pth')
                try:
                    torch.save(model.state_dict(), save_name, _use_new_zipfile_serialization=False)
                except:
                    print('Your version of Torch is lower than 1.7.0 !')
                    torch.save(model.state_dict(), save_name)

            if tblogger is not None:
                if dataset == 'voc':
                    tblogger.add_scalar('07test/mAP', evaluator.map, epoch)
                elif dataset == 'coco':
                    tblogger.add_scalar('val/AP50_95', evaluator.ap50_95, epoch)
                    tblogger.add_scalar('val/AP50', evaluator.ap50, epoch)

    if ddp:
        # wait for all processes to synchronize
        dist.barrier()

    # set train mode
    model.trainable = True
    model.set_grid(train_size)
    model.train()

    return best_map


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

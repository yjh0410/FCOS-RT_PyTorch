import argparse
import os

import torch

from evaluator.voc_evaluator import VOCAPIEvaluator
from evaluator.coco_evaluator import COCOAPIEvaluator

from data.transforms import ValTransforms

from utils.misc import TestTimeAugmentation


parser = argparse.ArgumentParser(description='FCOS-RT Evaluation')
# basic
parser.add_argument('-size', '--img_size', default=640, type=int,
                    help='img_size')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use cuda')
# model
parser.add_argument('-v', '--version', default='fcos_rt',
                    help='fcos_rt')
parser.add_argument('--trained_model', type=str,
                    default='weights/', 
                    help='Trained state_dict file path to open')
parser.add_argument('--conf_thresh', default=0.001, type=float,
                    help='NMS threshold')
parser.add_argument('--nms_thresh', default=0.6, type=float,
                    help='NMS threshold')
# dataset
parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                    help='data root')
parser.add_argument('-d', '--dataset', default='coco-val',
                    help='voc, coco-val, coco-test.')
# TTA
parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                    help='use test augmentation.')

args = parser.parse_args()


def voc_test(model, data_dir, device, img_size):
    evaluator = VOCAPIEvaluator(data_root=data_dir,
                                img_size=img_size,
                                device=device,
                                transform=ValTransforms(img_size),
                                display=True
                                )

    # VOC evaluation
    evaluator.evaluate(model)


def coco_test(model, data_dir, device, img_size, test=False):
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=img_size,
                        device=device,
                        testset=True,
                        transform=ValTransforms(img_size)
                        )

    else:
        # eval
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=img_size,
                        device=device,
                        testset=False,
                        transform=ValTransforms(img_size)
                        )

    # COCO evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    # dataset
    if args.dataset == 'voc':
        print('eval on voc ...')
        num_classes = 20
        data_dir = os.path.join(args.root, 'VOCdevkit')
    elif args.dataset == 'coco-val':
        print('eval on coco-val ...')
        num_classes = 80
        data_dir = os.path.join(args.root, 'COCO')
    elif args.dataset == 'coco-test':
        print('eval on coco-test-dev ...')
        num_classes = 80
        data_dir = os.path.join(args.root, 'COCO')
    else:
        print('unknow dataset !! we only support voc, coco-val, coco-test !!!')
        exit(0)

    # cuda
    if args.cuda:
        print('use cuda')
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # input size
    input_size = args.input_size

    # model
    model_name = args.version
    print('Model: ', model_name)

    # load model and config file
    if model_name == 'fcos_rt':
        from models.fcos_rt import FCOS_RT
        backbone = args.backbone
        # model
        model = FCOS_RT(device=device, 
                        img_size=input_size, 
                        num_classes=num_classes, 
                        trainable=False, 
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        bk=backbone)
    else:
        print('Unknown model name...')
        exit(0)


    # load weight
    model.load_state_dict(torch.load(args.trained_model, map_location=device))
    model.to(device).eval()
    print('Finished loading model!')

    # TTA
    test_aug = TestTimeAugmentation(num_classes=num_classes) if args.test_aug else None
    
    # evaluation
    with torch.no_grad():
        if args.dataset == 'voc':
            voc_test(model, data_dir, device, args.img_size)
        elif args.dataset == 'coco-val':
            coco_test(model, data_dir, device, args.img_size, test=False)
        elif args.dataset == 'coco-test':
            coco_test(model, data_dir, device, args.img_size, test=True)

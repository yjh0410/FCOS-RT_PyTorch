import torch
import argparse

from data import *

from utils.voc_evaluator import VOCAPIEvaluator
from utils.coco_evaluator import COCOAPIEvaluator


parser = argparse.ArgumentParser(description='FCOS-RT Evaluation')
parser.add_argument('-v', '--version', default='fcos_rt',
                    help='fcos_rt.')
parser.add_argument('-bk', '--backbone', default='r18',
                    help='r18, r50, r101')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val.')
parser.add_argument('-size', '--input_size', default=640, type=int,
                    help='input_size')
parser.add_argument('--trained_model', type=str,
                    default='weights/', 
                    help='Trained state_dict file path to open')
parser.add_argument('-ct', '--conf_thresh', default=0.001, type=float,
                    help='conf thresh')
parser.add_argument('-nt', '--nms_thresh', default=0.60, type=float,
                    help='nms thresh')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use cuda')

args = parser.parse_args()



def voc_test(model, device, input_size):
    evaluator = VOCAPIEvaluator(data_root=VOC_ROOT,
                                img_size=input_size,
                                device=device,
                                transform=BaseTransform(input_size),
                                labelmap=VOC_CLASSES,
                                display=True
                                )

    # VOC evaluation
    evaluator.evaluate(model)


def coco_test(model, device, input_size, test=False):
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
                        data_dir=coco_root,
                        img_size=input_size,
                        device=device,
                        testset=True,
                        transform=BaseTransform(input_size)
                        )

    else:
        # eval
        evaluator = COCOAPIEvaluator(
                        data_dir=coco_root,
                        img_size=input_size,
                        device=device,
                        testset=False,
                        transform=BaseTransform(input_size)
                        )

    # COCO evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    # dataset
    if args.dataset == 'voc':
        print('eval on voc ...')
        num_classes = 20
    elif args.dataset == 'coco-val':
        print('eval on coco-val ...')
        num_classes = 80
    elif args.dataset == 'coco-test':
        print('eval on coco-test-dev ...')
        num_classes = 80
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

    else:
        print('Unknown model name...')
        exit(0)

    # model
    net = FCOS_RT(device=device, 
                 img_size=input_size, 
                 num_classes=num_classes, 
                 trainable=False, 
                 conf_thresh=args.conf_thresh,
                 nms_thresh=args.nms_thresh,
                 bk=backbone
                 )

    # load weight
    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.to(device).eval()
    print('Finished loading model!')
    
    # evaluation
    with torch.no_grad():
        if args.dataset == 'voc':
            voc_test(net, device, input_size)
        elif args.dataset == 'coco-val':
            coco_test(net, device, input_size, test=False)
        elif args.dataset == 'coco-test':
            coco_test(net, device, input_size, test=True)

"""
Author: Haoxi Ran
Date: 05/10/2022
"""

from functools import partial

import argparse
import numpy as np
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import datetime
import logging
from pathlib import Path

from dataset.SanLimDataLoader import SanLimDataLoader
from modules.ptaug_utils import transform_point_cloud, scale_point_cloud, get_aug_args
from modules.pointnet2_utils import sample

from utils.utils import get_model, get_loss, set_seed, weight_init

def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('RepSurf')
    # Basic
    parser.add_argument('--data_dir', type=str, default='/data', help='data dir')
    parser.add_argument('--log_root', type=str, default='/sanlim_project/snapshot', help='log root dir')
    parser.add_argument('--model', default='repsurf.scanobjectnn.repsurf_ssg_umb',
                        help='model file name [default: repsurf_ssg_umb]')
    parser.add_argument('--seed', type=int, default=2800, help='Training Seed')
    parser.add_argument('--cuda_ops', action='store_true', default=False,
                        help='Whether to use cuda version operations [default: False]')

    # Training
    parser.add_argument('--batch_size', type=int, default=256, help='batch size in training [default: 64]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [Adam, SGD]')
    parser.add_argument('--scheduler', type=str, default='step', help='scheduler for training')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--decay_step', default=10, type=int, help='number of epoch per decay [default: 20]')
    parser.add_argument('--n_workers', type=int, default=12, help='DataLoader Workers Number [default: 1024]')
    parser.add_argument('--init', type=str, default=None, help='initializer for model [kaiming, xavier]')

    # Evaluation
    parser.add_argument('--min_val', type=int, default=0, help='Min val epoch [default: 0]')

    # Augmentation
    parser.add_argument('--aug_scale', action='store_true', default=False,
                        help='Whether to augment by scaling [default: False]')
    parser.add_argument('--aug_shift', action='store_true', default=False,
                        help='Whether to augment by shifting [default: False]')

    # Modeling
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--return_dist', action='store_true', default=True,
                        help='Whether to use signed distance [default: False]')
    parser.add_argument('--return_center', action='store_true', default=True,
                        help='Whether to return center in surface abstraction [default: False]')
    parser.add_argument('--return_polar', action='store_true', default=True,
                        help='Whether to return polar coordinate in surface abstraction [default: False]')
    parser.add_argument('--group_size', type=int, default=8, help='Size of umbrella group [default: 0]')
    parser.add_argument('--umb_pool', type=str, default='sum', help='pooling for umbrella repsurf [mean, max]')

    return parser.parse_args()


def test(model, loader, log_dir, num_class=3, num_point=1024, total_num=1, logger=None):
    correct = 0
    classifier = model.eval()
    pred_dict = {}
    threshold = 0.0
    
    for j, data in enumerate(loader):
        percent = 100*j*loader.batch_size/len(loader.dataset)
        if percent >= threshold:
            print('{:.2f}% done'.format(percent))
            threshold += 10.0
        points, target, name = data
        points, target = points.cuda(), target.cuda()

        # preprocess
        points = sample(num_point, points)
        
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct += pred_choice.eq(target.long().data).cpu().sum()
        
        # added
        pred_list = pred_choice.tolist()
        
        for k in range(len(name)):
            key = '_'.join(name[k].split('_')[:-1])
            if key not in pred_dict:
                pred_dict[key] = [0, 0, 0]
            pred_dict[key][pred_list[k]] += 1
    
    cnt = 0
    with open(os.path.join(log_dir, 'best.txt'), 'w') as f:
        for key in pred_dict.keys():
            classname = key.split('_')[0]
            corr = loader.dataset.classes[classname]
            guess = pred_dict[key].index(max(pred_dict[key]))
            if guess == corr: cnt += 1
            logger.info('{}: {}, {}, {}'.format(key, *pred_dict[key]))
            f.write('{}: {}, {}, {}\n'.format(key, *pred_dict[key]))
        
    total_num = len(pred_dict)
    acc = cnt / total_num

    return acc


def main(args):
    def log_string(s):
        logger.info(s)
        print(s)

    '''HYPER PARAMETER'''
    set_seed(args.seed)

    '''CREATE DIR'''
    experiment_dir = Path(args.log_root)
    log_dir = 'RepSurf_{}_np{}_bs{}_lr{}_dr{}_ds{}'\
        .format(args.optimizer, args.num_point, args.batch_size, args.learning_rate, args.decay_rate, args.decay_step)
    experiment_dir = experiment_dir.joinpath(log_dir)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    log_dir = experiment_dir.joinpath('logs/')

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s_test.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    args.num_class = 3
    args.dataset = 'SanLim'
    TEST_DATASET = SanLimDataLoader(root=args.data_dir, split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.n_workers)

    '''MODEL BUILDING'''
    print(str(experiment_dir))
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    epoch = checkpoint['epoch']
    classifier = torch.nn.DataParallel(checkpoint['model']).cuda()
    log_string('Test model from {:d} epoch'.format(epoch))

    #classifier = torch.nn.DataParallel(get_model(args)).cuda()
    #classifier.load_state_dict(checkpoint['model_state_dict'])

    '''OPTIMIZER'''
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=args.learning_rate,
            momentum=0.9)

    '''LR SCHEDULER'''
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=0.7)
    else:
        raise Exception('No Such Scheduler')
    
    with torch.no_grad():
        acc = test(classifier.eval(), testDataLoader, log_dir, num_class=args.num_class, num_point=args.num_point,
                                    total_num=len(TEST_DATASET), logger=logger)

        log_string('Test Accuracy: %.2f' % (acc * 100))


if __name__ == '__main__':
    args = parse_args()
    main(args)

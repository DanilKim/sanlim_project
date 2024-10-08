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


def test(model, loader, num_class=3, num_point=1024, num_votes=1, total_num=1, logger=None):
    vote_correct = 0
    sing_correct = 0
    classifier = model.eval()
    sing_pred_dict = {}
    vote_pred_dict = {}

    for j, data in enumerate(loader):
        points, target, name = data
        points, target = points.cuda(), target.cuda()

        # preprocess
        points = sample(num_point, points)

        # vote
        vote_pool = torch.zeros(target.shape[0], num_class).cuda()
        for i in range(num_votes):
            new_points = points.clone()
            # scale
            if i > 0:
                new_points[:, :3] = scale_point_cloud(new_points[:, :3])
            # predict
            pred = classifier(new_points)
            # single
            if i == 0:
                sing_pred = pred
            # vote
            vote_pool += pred
        vote_pred = vote_pool / num_votes

        # single pred
        sing_pred_choice = sing_pred.data.max(1)[1]
        sing_correct += sing_pred_choice.eq(target.long().data).cpu().sum()
        # vote pred
        vote_pred_choice = vote_pred.data.max(1)[1]
        vote_correct += vote_pred_choice.eq(target.long().data).cpu().sum()
        
        # added
        sing_list = sing_pred_choice.tolist()
        vote_list = vote_pred_choice.tolist()
        
        for j in range(len(name)):
            key = '_'.join(name[j].split('_')[:-1])
            if key not in sing_pred_dict:
                sing_pred_dict[key] = [0, 0, 0]
            if key not in vote_pred_dict:
                vote_pred_dict[key] = [0, 0, 0]
            sing_pred_dict[key][sing_list[j]] += 1
            vote_pred_dict[key][vote_list[j]] += 1
    
    sing_cnt = 0
    vote_cnt = 0
    sing_err = []
    vote_err = []
    
    for key in sing_pred_dict.keys():
        classname = key.split('_')[0]
        corr = loader.dataset.classes[classname]
        sing = sing_pred_dict[key].index(max(sing_pred_dict[key]))
        vote = vote_pred_dict[key].index(max(vote_pred_dict[key]))
        if sing == corr:
            sing_cnt += 1
        else:
            sing_err.append(key)
        if vote == corr:
            vote_cnt += 1
        else:
            vote_err.append(key)
        print('{}: {}, {}, {}'.format(key, *sing_pred_dict[key]))
        # print('{}: {}, {}, {}'.format(key, *vote_pred_dict[key]))
    
    # print('----single prediction error list----')
    # for key in sing_err:
    #     print('{}: {}, {}, {}'.format(key, *sing_pred_dict[key]))
    # print('----vote prediction error list----')
    # for key in vote_err:
    #     print('{}: {}, {}, {}'.format(key, *vote_pred_dict[key]))
        
    total_num = len(sing_pred_dict)
    sing_acc = sing_cnt / total_num
    vote_acc = vote_cnt / total_num
    
    # sing_acc = sing_correct.item() / total_num
    # vote_acc = vote_correct.item() / total_num

    return sing_acc, vote_acc


def main(args):
    def log_string(s):
        logger.info(s)
        print(s)

    '''HYPER PARAMETER'''
    set_seed(args.seed)

    '''CREATE DIR'''
    experiment_dir = Path(args.log_root)
    experiment_dir.mkdir(exist_ok=True)
    log_dir = 'RepSurf_{}_np{}_bs{}_lr{}_dr{}_ds{}'\
        .format(args.optimizer, args.num_point, args.batch_size, args.learning_rate, args.decay_rate, args.decay_step)
    experiment_dir = experiment_dir.joinpath(log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    args.num_class = 3
    args.dataset = 'SanLim'
    TRAIN_DATASET = SanLimDataLoader(root=args.data_dir, split='train')
    VALID_DATASET = SanLimDataLoader(root=args.data_dir, split='val')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.n_workers, drop_last=True)
    validDataLoader = torch.utils.data.DataLoader(VALID_DATASET, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.n_workers)

    '''MODEL BUILDING'''
    classifier = torch.nn.DataParallel(get_model(args)).cuda()
    criterion = get_loss().cuda()

    try:
        print(str(experiment_dir))
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier = checkpoint['model']
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        if args.init:
            init_func = partial(weight_init, init_type=args.init)
            classifier = classifier.apply(init_func)

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

    global_epoch = 0
    global_step = 0
    best_sing_acc = 0.0
    best_vote_acc = 0.0
    loader_len = len(trainDataLoader)

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        train_loss = []
        train_correct = 0

        scheduler.step()
        for batch_id, data in enumerate(trainDataLoader):
            '''INPUT'''
            points, target, _ = data
            points, target = points.cuda(), target.cuda()

            '''PREPROCESS'''
            points = sample(args.num_point, points)
            # points = transform_point_cloud(points, args, aug_args)

            '''FORWARD'''
            optimizer.zero_grad()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            classifier = classifier.train()
            pred = classifier(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            train_correct += correct
            train_loss.append(loss.item())

            '''BACKWARD'''
            loss.backward()
            optimizer.step()
            global_step += 1

            if batch_id % 80 == 0:
                print('Epoch: [{0}][{1}/{2}] lr {lr:.6f} loss {loss:.4f}'.
                      format(epoch, batch_id, len(trainDataLoader), lr=lr, loss=loss.item()))

        train_instance_acc = train_correct.item() / (loader_len * args.batch_size)
        train_mean_loss = np.mean(train_loss)
        log_string('Train Instance Accuracy: %.2f, Loss: %f' % (train_instance_acc * 100, train_mean_loss))

        if epoch >= args.min_val:
            with torch.no_grad():
                sing_acc, vote_acc = test(classifier.eval(), validDataLoader, num_class=args.num_class, num_point=args.num_point,
                                          total_num=len(VALID_DATASET))

                if sing_acc >= best_sing_acc:
                    best_sing_acc = sing_acc
                if vote_acc >= best_vote_acc:
                    best_vote_acc = vote_acc
                    best_epoch = epoch + 1

                log_string('Test Single Accuracy: %.2f' % (sing_acc * 100))
                log_string('Best Single Accuracy: %.2f' % (best_sing_acc * 100))
                log_string('Test Vote Accuracy: %.2f' % (vote_acc * 100))
                log_string('Best Vote Accuracy: %.2f' % (best_vote_acc * 100))

                if vote_acc >= best_vote_acc:
                    logger.info('Save model...')
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': best_epoch,
                        'vote_acc': vote_acc,
                        'model': classifier,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
        global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)

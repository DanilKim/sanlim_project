import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TreePCDDataset
from optimizer import get_optimizer
import model as m

import warnings
warnings.filterwarnings('ignore')

from logger import set_logger
from utils import save_model, voting
import math
import pdb

def parse_arg():
    parser = argparse.ArgumentParser()

    ## path settings ##
    parser.add_argument('--list_root', type=str, default='./datalist')
    parser.add_argument('--data_root', type=str, default='/data')
    parser.add_argument('--snapshot_root', type=str, default='./snapshot')

    ## hyperparameters ##
    parser.add_argument('--K', type=int, default=6)
    parser.add_argument('--L', type=int, default=8)
    parser.add_argument('--num_points', type=int, default=516, 
                        help='number of input points in a tree sample.')

    ## experiment settings ##
    parser.add_argument('--model', type=str, default='SurfG3D18', choices=['G3DNet18', 'SurfG3D18'])
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
    parser.add_argument('--resume', action='store_true', help='resume from latest epoch')
    parser.add_argument('--ckpt_path', type=str, default='', help='finetune from designated model checkpoint')
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_rate_step', type=int, default=0)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--print_every_iter', type=int, default=200, help='num iterations to log periodically')
    
    ## random ##
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


def load_model(path, device):
    try:
        checkpoint = torch.load(path, map_location=device)
        #model.load_state_dict(checkpoint['model_state_dict'])
        model = checkpoint['model']
        model.to(device)
        start_epoch, best_acc = checkpoint['epoch'], checkpoint['best_acc']
    except:
        print('Loading checkpoint from {} failed.. Learning from Scratch!'.format(path))
        start_epoch, best_acc = 1, 0
    return model, start_epoch, best_acc


def train(cfg, model, logger, train_loader, val_loader, criterion, optimizer, save_dir, val_every, device):
    print('Start training epoch {}'.format(cfg.start_epoch))
    best_acc = cfg.best_acc

    #### Training Epochs ####
    for epoch in range(cfg.start_epoch, cfg.max_epoch):
        logger.info('Starting {} epoch out of {} epochs'.format(epoch, cfg.max_epoch))
        model.train()

        #### Training Iterations ####
        num_samples = 0
        correct_predictions = 0
        for i, (V, A, L, P, _) in enumerate(train_loader):
            V, A, L = V.to(device), A.to(device), L.to(device)
            P = [p.to(device) for p in P]

            output = model(V, A, P)
            loss = criterion(output, L)
            if math.isnan(loss):
                print('Nan!')
                pdb.set_trace()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred = torch.argmax(output, dim=1).detach()
            num_samples += float(pred.size(0)) 
            correct_predictions += (pred == L).float().cpu().sum().item()
            
            if i % cfg.print_every_iter == 0:
                acc = 1.0 * correct_predictions / num_samples
                logger.info('[Train Epoch {}] {:.1f}% done. loss: {:.4f} / acc: {:.2f}%'.format(
                    epoch, 100 * float(i+1) / len(train_loader), loss, 100 * acc))

        #### Evaluate Train & Val Performance ####
        train_acc = 1.0 * correct_predictions / num_samples
        logger.info('Training Epoch {} finished. Average Acc : {:.3f}%\n'.format(epoch, 100 * train_acc))
        val_acc = evaluation(cfg, epoch, model, logger, val_loader, device)
        logger.info('Evaluation Epoch {} finished. Average Acc : {:.3f}%'.format(epoch, 100 * val_acc))

        #### Renew Best Model ####
        if val_acc > best_acc:
            best_acc = val_acc
            logger.info('Best Accuracy @ Epoch {}, Acc {:.3f}%! '.format(epoch, 100 * best_acc))
            save_fn = os.path.join(save_dir, 'checkpoints/best.pt')
            print('Saving Best Model to {}'.format(save_fn))
            save_model(model, epoch, best_acc, save_fn)

        #### Save Single Epoch Model ####
        save_fn = os.path.join(save_dir, 'checkpoints/latest.pt')
        print('Saving Latest Model @ Epoch {} to {} ...\n'.format(epoch, save_fn))
        save_model(model, epoch, best_acc, save_fn)

        #### Drop Learning Rate ####
        if cfg.learning_rate_step > 0 and (epoch) % cfg.learning_rate_step == 0: 
            optimizer.param_groups[0]['lr'] *= 0.9
            logger.info('Learning rate dropped to {} from epoch {}'.format(
                optimizer.param_groups[0]['lr'], epoch))


def evaluation(cfg, epoch, model, logger, val_loader, device):
    logger.info('Start evaluating Epoch {}.'.format(epoch))
    model.eval()

    with torch.no_grad():
        #### Evaluation Iterations ####
        votes, gts = {}, {}
        correct_predictions = 0
        for i, (V, A, L, P, names) in enumerate(val_loader):
            V, A = V.to(device), A.to(device)
            P = [p.to(device) for p in P]
            output = model(V, A, P)

            pred = torch.argmax(output, dim=1).detach().cpu().tolist()
            labels = L.tolist()
            votes, gts = voting(pred, labels, names, votes, gts)

            if i % cfg.print_every_iter == 0:
                logger.info('[Eval Epoch {}] {:.1f}% done. '.format(
                    epoch, 100 * float(i+1) / len(val_loader)))
        
        #### Per-tree evaluation by voting ####
        for name in votes:
            print(name, votes[name], gts[name])
            if np.argmax(votes[name]) == gts[name]:
                correct_predictions += 1
        
        val_acc = float(correct_predictions) / len(votes)
    return val_acc


def main():
    args = parse_arg()

    exp_name = '{}_{}_np{}_bs{}_lr{}_lrs{}_wd{}'.format(
        args.model, args.optimizer, args.num_points, args.batch_size, 
        args.learning_rate, args.learning_rate_step, args.l2
    )

    save_dir = os.path.join(args.snapshot_root, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    print('Start Experiment {}.'.format(exp_name))
    print('Checkpoints and logs are saved in {}'.format(save_dir))
    
    print('Configs are:')
    print('{}'.format(args))

    print('Setting random seed & CUDA Device...')
    print('GPU Availability: {} / #GPUS {}\n'.format(torch.cuda.is_available(), torch.cuda.device_count()))

    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Point Cloud data from {}".format(args))
    train_dataset = TreePCDDataset(args, split='train', mode='train')
    val_dataset = TreePCDDataset(args, split='val', mode='eval')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=4)

    print()
    print('Defining GraphCNN Network, Loss, Optimizer...\n')
    model = getattr(m, args.model)(args.num_points, args.L, 3, 0)
    model.to(device)
    if args.ckpt_path != '':
        model, args.start_epoch, args.best_acc = load_model(args.ckpt_path, device)
    elif args.resume:
        model, args.start_epoch, args.best_acc = load_model(os.path.join(save_dir, 'checkpoints/latest.pt'), device)
    else:
        args.start_epoch, args.best_acc = 1, 0

    loss = nn.CrossEntropyLoss()
    if args.learning_rate_step > 0:
        args.learning_rate *= ((0.9) ** ((args.start_epoch-1) // args.learning_rate_step))
    optimizer = get_optimizer(name=args.optimizer, params=model.parameters(), lr=args.learning_rate, weight_decay=args.l2)

    log_dir = os.path.join(save_dir, 'logs')    
    logger = set_logger(log_dir)

    train(args, model, logger, train_loader, val_loader, loss, optimizer, save_dir, 1, device)


if __name__ == "__main__":
    main()



    
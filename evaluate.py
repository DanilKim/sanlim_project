import os
import argparse
import random
import numpy as np
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TreePCDDataset
import model as m

import warnings
warnings.filterwarnings('ignore')

from logger import set_logger
from utils import save_model, voting

def parse_arg():
    parser = argparse.ArgumentParser()

    ## select mode ##
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'feat'])

    ## path settings ##
    parser.add_argument('--list_root', type=str, default='./datalist')
    parser.add_argument('--data_root', type=str, default='/data')
    parser.add_argument('--snapshot_root', type=str, default='/data/snapshot')
    parser.add_argument('--summary_root', type=str, default='/data/summary')
    parser.add_argument('--split', type=str, default='test')

    ## hyperparameters ##
    parser.add_argument('--K', type=int, default=6)
    parser.add_argument('--L', type=int, default=8)
    parser.add_argument('--num_points', type=int, default=516, 
                        help='number of input points in a tree sample.')
    parser.add_argument('--subsample_W', type=float, default=0.01, 
                        help='grid subsampling width. set to 0 if not subsampled')

    ## experiment settings ##
    parser.add_argument('--model', type=str, default='SurfG3D18', choices=['G3DNet18', 'SurfG3D18'])
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
    parser.add_argument('--ver', type=str, default='best', choices=['best', 'latest'], help='eval model trained until this epoch')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_rate_step', type=int, default=0)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--no_aug', action='store_true')
    parser.add_argument('--print_every_iter', type=int, default=50, help='num iterations to print periodically')
    
    ## random ##
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def evaluate(cfg, epoch, model, eval_loader, save_dir, device):
    print('Start evaluating Epoch ' + epoch)
    model.eval()

    label_map = eval_loader.dataset.inverse_label_map
    right_ans = {True: 'O', False: 'X'}

    result_dir = os.path.join(save_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)
    result_fn = os.path.join(result_dir, cfg.ver+'.csv')

    with torch.no_grad():
        #### Evaluation Iterations ####
        votes, gts = {}, {}
        correct_predictions = 0
        for i, (V, A, L, P, names) in enumerate(eval_loader):
            V, A = V.to(device), A.to(device)
            P = [p.to(device) for p in P]
            output = model(V, A, P)

            pred = torch.argmax(output, dim=1).detach().cpu().tolist()
            labels = L.tolist()
            votes, gts = voting(pred, labels, names, votes, gts)

            if i % cfg.print_every_iter == 0:
                print('[ Eval {} Epoch ] on {} split {:.1f}% done. '.format(
                    epoch, cfg.split, 100 * float(i+1) / len(eval_loader)))     
        
        #### Per-tree evaluation by voting ####
        with open(result_fn, 'w', encoding='utf-8-sig') as rf:
            wr = csv.DictWriter(rf, delimiter=',', fieldnames=[
                '수명', '예측결과', '실제구분', '정답여부', '투표:칩엽수', '투표:활엽수', '투표:기타수종'
            ])
            wr.writeheader()
            for name in votes:
                print(name, votes[name], gts[name])
                wr.writerow({
                    '수명': name,
                    '예측결과': label_map[np.argmax(votes[name])],
                    '실제구분': label_map[gts[name]],
                    '정답여부': right_ans[np.argmax(votes[name]) == gts[name]],
                    '투표:칩엽수': int(votes[name][0]),
                    '투표:활엽수': int(votes[name][1]), 
                    '투표:기타수종': int(votes[name][2])
                })
                if np.argmax(votes[name]) == gts[name]:
                    correct_predictions += 1

        val_acc = 100 * float(correct_predictions) / len(votes)
        print('Overall Accuracy : {:.2f}'.format(val_acc))


def extract_feature(cfg, epoch, model, eval_loader, save_dir, device):
    print('Start extractring features from Epoch ' + epoch)
    model.eval()

    result_dir = os.path.join(save_dir, 'features')
    os.makedirs(result_dir, exist_ok=True)
    result_fn = os.path.join(result_dir, cfg.ver+'_'+epoch+'.npy')

    with torch.no_grad():
        #### Evaluation Iterations ####
        features = []
        for i, (V, A, _, P, _) in enumerate(eval_loader):
            V, A = V.to(device), A.to(device)
            P = [p.to(device) for p in P]
            V, A = model.Block_128(V, A, P)
            V, A = model.Block_256(V, A, P)
            V, A = model.Block_512(V, A, P)
            V, A = model.Block_1024(V, A, P)
            logits = model.flatten(V)
            logits = model.fc_2048(logits)
            logits = model.bn(logits)
            logits = model.relu(logits)

            features.append(logits.cpu().numpy())

            if i % cfg.print_every_iter == 0:
                print('Extracting fc1 fearture for {} epoch {:.1f}% done. '.format(
                    epoch, 100 * float(i+1) / len(eval_loader)))
                print(logits.shape)
        
        features = np.concatenate(features, axis=0)
        np.save(result_fn, features)
        print('Extracted feature saved to {}'.format(result_fn))


def main():
    args = parse_arg()

    aug = '_noaug' if args.no_aug else ''
    dropout = '_do' + str(args.dropout) if args.dropout > 0 else ''
    exp_name = '{}_{}_np{}_bs{}_lr{}_lrs{}_wd{}{}{}'.format(
        args.model, args.optimizer, args.num_points, args.batch_size, 
        args.learning_rate, args.learning_rate_step, args.l2, dropout, aug
    )
    save_dir = os.path.join(args.snapshot_root, exp_name)

    print('Evaluate Experiment {} for the {} epoch.'.format(exp_name, args.ver))

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
    eval_dataset = TreePCDDataset(args, split=args.split, mode='eval')
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=4)

    print()
    print('Defining GraphCNN Network, Loss, Optimizer...\n')
    #model = getattr(m, args.model)(args.num_points, args.L, 3, args.dropout)

    model_path = os.path.join(save_dir, 'checkpoints/{}.pt'.format(args.ver))
    checkpoint = torch.load(model_path, map_location=device)
    #model.load_state_dict(checkpoint)
    #model.load_state_dict(checkpoint['model_state_dict'])
    model = checkpoint['model']
    model.to(device)
    ep = str(checkpoint['epoch'])

    if args.mode == 'eval':
        evaluate(args, ep, model, eval_loader, save_dir, device)
    else:
        extract_feature(args, ep, model, eval_loader, save_dir, device)

if __name__ == "__main__":
    main()



    
import os
import argparse
import csv
from logger import set_logger
from labelmap import label_to_kind
import pdb

def parse_arg():
    parser = argparse.ArgumentParser()

    ## result path ##
    parser.add_argument('--g3dnet', type=str, required=True)
    parser.add_argument('--repsurf', type=str, required=True)
    parser.add_argument('--snapshot_dir', type=str, default='/data/snapshot')
    return parser.parse_args()

def main(args, logger, save_dir):
  logger.info('Start testing G3DNet18 + RepSurf model ensemble')

  kind_label_map, label_map, inverse_label_map = label_to_kind()
  label_dict = {}
  classes = {k: label_map[v] for k, v in kind_label_map.items()}
  right_ans = {True: 'O', False: 'X'}
      
  g3dnet = {}
  with open(os.path.join(args.snapshot_dir, args.g3dnet, 'results/best.csv'), 'r', encoding='utf-8-sig') as f:
    line = f.readline()
    while True:
      line = f.readline()
      if not line: break
      name, _, _ , _, zero, one, two = line.split(',')
      g3dnet[name] = (int(zero), int(one), int(two))
      label_dict[name] = classes[name.split('_')[0]]
  num_samples = len(label_dict.keys())

  repsurf = {}
  with open(os.path.join(args.snapshot_dir, args.repsurf, 'logs/best.txt'), 'r') as g:
    while True:
      line = g.readline()
      if not line: break
      name, num = line.split(':')
      zero, one, two = map(int, num.split(','))
      repsurf[name] = (zero, one, two)

  
  cnt = {'g3dnet': 0, 'repsurf': 0, 'ensemble': 0}
  #TF = {True: 0, False:0}
  with open(os.path.join(save_dir, 'result.csv'), 'w', encoding='utf-8-sig') as rf:
    wr = csv.DictWriter(rf, delimiter=',', fieldnames=[
        '파일명', '예측결과', '실제구분', '정답여부', '투표:칩엽수', '투표:활엽수', '투표:기타수종'
    ])
    wr.writeheader()
    logger.info('파일명 : G3DNet18 정답 여부 | Repsurf 정답 여부 | Ensemble 정답 여부 ')
    for s in g3dnet:
      if s in repsurf:
        g3dnet_result = g3dnet[s].index(max(g3dnet[s]))
        repsurf_result = repsurf[s].index(max(repsurf[s]))
        ensemble = [g + r for g, r in zip(g3dnet[s], repsurf[s])]
        ensemble_result = ensemble.index(max(ensemble))

        wr.writerow({
            '파일명': name,
            '예측결과': inverse_label_map[ensemble_result],
            '실제구분': inverse_label_map[label_dict[s]],
            '정답여부': right_ans[ensemble_result == label_dict[s]],
        })

        g3dnet_result = (g3dnet_result == label_dict[s])
        repsurf_result = (repsurf_result == label_dict[s])
        ensemble_result = (ensemble_result == label_dict[s])
        if g3dnet_result: cnt['g3dnet'] += 1
        if repsurf_result: cnt['repsurf'] += 1
        if ensemble_result: cnt['ensemble'] += 1
        print('{}: {}, {}, {}'.format(s, g3dnet_result, repsurf_result, ensemble_result, label_dict[s]))
        logger.info('{}:      {}      |      {}      |      {}     '.format(
            s, g3dnet_result, repsurf_result, ensemble_result, label_dict[s]
        ))

    logger.info('--------- 성능 평가 ---------')
    #logger.info('TP : {} | FP : {} | TN : {} | FN : {}'.format())
    logger.info('Overall Accuracy (OA) : {:.2f}%'.format(100 * cnt['ensemble'] / num_samples))



if __name__ == "__main__":
  args = parse_arg()

  save_dir = os.path.join(args.snapshot_dir, 'ensemble')
  os.makedirs(save_dir, exist_ok=True)
  logger = set_logger(save_dir)

  logger.info('+++++++++++++++++++++++++++++++')
  main(args, logger, save_dir)

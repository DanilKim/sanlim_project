

label_dict = {}
classes = {
  'AC': 1, 'AG': 1, 'AH': 0, 'AP': 1, 'CA': 1, 'CC': 1, 'CEJ': 1, 'CHP': 0,
  'CK': 1, 'CMP': 1, 'CO': 1, 'CP': 0, 'STJ': 1, 'PR': 0,
  'QQ': 1, 'LL': 0, 'TB': 0, 'GB': 0, 'ZS': 1, 'QV': 1, 'MP': 1, 'PK': 0, 'PD': 0,
  'AA': 1, 'AK': 0, 'AN': 1, 'APY': 1, 'AVE': 0, 'CHAP': 0, 'CJ': 0, 'COK': 1,
  'CPV': 0, 'DIK': 1, 'FR': 1, 'HPD': 0, 'JU': 0, 'KP': 1, 
  'MAG': 1, 'PAU': 1, 'PC': 1, 'PDM': 0, 'PIS': 0, 
  'PLO': 0, 'PTA': 1, 'QA': 1, 'QS': 1, 'QV': 1, 'SA': 1, 'SP': 1,
  'STR': 0, 'UDV': 1, 'ZZ': 1, 'ALJ': 0, 'BP': 1, 'CAJ': 1,
  'CD': 0, 'CES': 1, 'CR': 1, 'EA': 1, 'JR': 1, 'KAS': 1, 'LI': 1,
  'LT': 1, 'MG': 0, 'PA': 0, 'PB': 0, 'PE': 0, 'PPU': 0, 'PS': 1, 'PY': 1, 
  'QUA': 1, 'RP': 1, 'SJ': 1, 'STY': 1, 'TX': 0, 'ZS': 1, 'AT': 1,
  'QY': 1, 'HE': 1, 'FS': 1, 'Ul': 1, 'AHI': 1, 'UP': 1, 'EU': 1,
  'PQ': 1, 'ABN': 0, 'CPP': 1, 'AJ': 1, 'BD': 1, 'SOA': 1, 'SO': 1,
  'COR': 1, 'CAO': 1, 'QD': 1, 'QM': 1, 'HD': 1,
  'TD': 1, 'CAT': 1, 'SB': 1, 'POD': 1, 'ACM': 1,
  'CL': 1, 'TI': 1, 'IP': 1, 'AM': 1, 'CU': 0, 'BB': 2, 
  'JM': 1, 'ACD': 1, 'CB': 1, 'AE': 1, 'PO': 1, 'DM': 1, 
  'FM': 1, 'TV': 1, 'MAU': 1, 'PSE': 1, 'IC': 1,
  'BS': 1, 'POC': 1, 'TN': 0, 'PHA': 1, 'BK': 1, 'PRS': 1, 'TKM':1,
  'STO': 1, 'STP': 1, 'HC': 1, 'MT': 1, 'VA': 1, 'TSM': 1, 'CN': 1,
  'EJ': 1, 'TS': 1
}
    
g3dnet = {}
with open('./best_41_test.csv', 'r') as f:
  while True:
    line = f.readline()
    if not line: break
    name, zero, one, two = line.split(',')
    g3dnet[name] = (int(zero), int(one), int(two))
    label_dict[name] = classes[name.split('_')[0]]


repsurf = {}
with open('./best.txt', 'r') as g:
  while True:
    line = g.readline()
    if not line: break
    name, num = line.split(':')
    zero, one, two = map(int, num.split(','))
    repsurf[name] = (zero, one, two)


cnt = {'g3dnet': 0, 'repsurf': 0, 'ensemble': 0}
d = {}
for s in g3dnet:
  if s in repsurf:
    g3dnet_result = g3dnet[s].index(max(g3dnet[s]))
    repsurf_result = repsurf[s].index(max(repsurf[s]))
    ensemble = [g + r for g, r in zip(g3dnet[s], repsurf[s])]
    ensemble_result = ensemble.index(max(ensemble))
    g3dnet_result = (g3dnet_result == label_dict[s])
    repsurf_result = (repsurf_result == label_dict[s])
    ensemble_result = (ensemble_result == label_dict[s])
    if g3dnet_result: cnt['g3dnet'] += 1
    if repsurf_result: cnt['repsurf'] += 1
    if ensemble_result: cnt['ensemble'] += 1
    print('{}: {}, {}, {}'.format(s, g3dnet_result, repsurf_result, ensemble_result, label_dict[s]))
    try:
      d[(g3dnet_result, repsurf_result, ensemble_result)] += 1
    except:
      d[(g3dnet_result, repsurf_result, ensemble_result)] = 1
    
print(d)
print(cnt)


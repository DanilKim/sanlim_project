def label_to_kind():

    label2kind = {
        'AC': 'bl', 'AG': 'bl', 'AH': 'nl', 'AP': 'bl', 'CA': 'bl', 'CC': 'bl', 'CEJ': 'bl', 'CHP': 'nl',
        'CK': 'bl', 'CMP': 'bl', 'CO': 'bl', 'CP': 'nl', 'STJ': 'bl', 'PR': 'nl',
        'QQ': 'bl', 'LL': 'nl', 'TB': 'nl', 'GB': 'nl', 'ZS': 'bl', 'QV': 'bl', 'MP': 'bl', 'PK': 'nl', 'PD': 'nl',
        'AA': 'bl', 'AK': 'nl', 'AN': 'bl', 'APY': 'bl', 'AVE': 'nl', 'CHAP': 'nl', 'CJ': 'nl', 'COK': 'bl',
        'CPV': 'nl', 'DIK': 'bl', 'FR': 'bl', 'HPD': 'nl', 'JU': 'nl', 'KP': 'bl', 
        'MAG': 'bl', 'PAU': 'bl', 'PC': 'bl', 'PDM': 'nl', 'PIS': 'nl', 
        'PLO': 'nl', 'PTA': 'bl', 'QA': 'bl', 'QS': 'bl', 'QV': 'bl', 'SA': 'bl', 'SP': 'nl',
        'STR': 'nl', 'UDV': 'bl', 'ZZ': 'bl', 'ALJ': 'nl', 'BP': 'bl', 'CAJ': 'bl',
        'CD': 'nl', 'CES': 'bl', 'CR': 'bl', 'EA': 'bl', 'JR': 'bl', 'KAS': 'bl', 'LI': 'bl',
        'LT': 'bl', 'MG': 'nl', 'PA': 'nl', 'PB': 'nl', 'PE': 'nl', 'PPU': 'nl', 'PS': 'bl', 'PY': 'bl', 
        'QUA': 'bl', 'RP': 'bl', 'SJ': 'bl', 'STY': 'bl', 'TX': 'nl', 'ZS': 'bl', 'AT': 'bl',
        'QY': 'bl', 'HE': 'bl', 'FS': 'bl', 'Ul': 'bl', 'AHI': 'bl', 'UP': 'bl', 'EU': 'bl',
        'PQ': 'bl', 'ABN': 'nl', 'CPP': 'bl', 'AJ': 'bl', 'BD': 'bl', 'SOA': 'bl', 'SO': 'bl',
        'COR': 'bl', 'CAO': 'bl', 'QD': 'bl', 'QM': 'bl', 'HD': 'bl',
        'TD': 'bl', 'CAT': 'bl', 'SB': 'bl', 'POD': 'bl', 'ACM': 'bl',
        'CL': 'bl', 'TI': 'bl', 'IP': 'bl', 'AM': 'bl', 'CU': 'nl', 'BB': 'bb', 
        'JM': 'bl', 'ACD': 'bl', 'CB': 'bl', 'AE': 'bl', 'PO': 'bl', 'DM': 'bl', 
        'FM': 'bl', 'TV': 'bl', 'MAU': 'bl', 'PSE': 'bl', 'IC': 'bl',
        'BS': 'bl', 'POC': 'bl', 'TN': 'nl', 'PHA': 'bl', 'BK': 'bl', 'PRS': 'bl', 'TKM':'bl',
        'STO': 'bl', 'STP': 'bl', 'HC': 'bl', 'MT': 'bl', 'VA': 'bl', 'TSM': 'bl', 'CN': 'bl',
        'TS': 'bl', 'EJ': 'bl'
    }

    label_map = {'nl':0, 'bl':1, 'bb':2}
    
    inverse_label_map = {0:'침엽수', 1:'활엽수', 2:'기타수종'}

    return label2kind, label_map, inverse_label_map
def label_to_kind():

    label2kind = {
        'AA': 'bl', 'ABN': 'nl', 'AC': 'bl', 'ACD': 'bl', 'ACM': 'bl', 'AE': 'bl', 'AG': 'bl', 'AH': 'nl', 
        'AHI': 'bl', 'AJ': 'bl', 'AK': 'nl', 'ALJ': 'nl', 'AM': 'bl', 'AN': 'bl', 'AP': 'bl', 'APY': 'bl', 
        'AT': 'bl', 'AVE': 'nl', 
        'BB': 'bb', 'BD': 'bl', 'BK': 'bl', 'BP': 'bl', 'BS': 'bl', 
        'CA': 'bl', 'CAJ': 'bl', 'CAO': 'bl', 'CAT': 'bl', 'CB': 'bl', 'CC': 'bl', 'CD': 'nl', 'CEJ': 'bl', 
        'CES': 'bl', 'CHAP': 'nl', 'CHP': 'nl', 'CJ': 'nl', 'CK': 'bl', 'CL': 'bl', 'CMP': 'bl', 'CN': 'bl', 
        'CO': 'bl', 'COK': 'bl', 'COR': 'bl', 'CP': 'nl', 'CPP': 'bl', 'CPV': 'nl', 'CR': 'bl', 'CU': 'nl', 
        'DIK': 'bl', 'DM': 'bl', 
        'EA': 'bl', 'EJ': 'bl', 'EU': 'bl', 
        'FM': 'bl', 'FR': 'bl', 'FS': 'bl', 
        'GB': 'nl', 
        'HC': 'bl', 'HD': 'bl', 'HE': 'bl', 'HPD': 'nl', 
        'IC': 'bl', 'IP': 'bl', 
        'JM': 'bl', 'JR': 'bl', 'JU': 'nl', 
        'KAS': 'bl', 'KP': 'bl', 
        'LI': 'bl', 'LL': 'nl', 'LT': 'bl', 
        'MAG': 'bl', 'MAU': 'bl', 'MG': 'nl', 'MP': 'bl', 'MT': 'bl', 
        'PA': 'nl', 'PAU': 'bl', 'PB': 'nl', 'PC': 'bl', 'PD': 'nl', 'PDM': 'nl', 'PE': 'nl', 'PHA': 'bl', 
        'PIS': 'nl', 'PK': 'nl', 'PLO': 'nl', 'PO': 'bl', 'POC': 'bl', 'POD': 'bl', 'PPU': 'nl', 'PQ': 'bl', 
        'PR': 'nl', 'PRS': 'bl', 'PS': 'bl', 'PSE': 'bl', 'PTA': 'bl', 'PY': 'bl', 
        'QA': 'bl', 'QD': 'bl', 'QM': 'bl', 'QQ': 'bl', 'QS': 'bl', 'QUA': 'bl', 'QV': 'bl', 'QY': 'bl', 
        'RP': 'bl', 
        'SA': 'bl', 'SB': 'bl', 'SJ': 'bl', 'SO': 'bl', 'SOA': 'bl', 'SP': 'bl', 'STJ': 'bl', 'STO': 'bl', 
        'STP': 'bl', 'STR': 'nl', 'STY': 'bl', 
        'TB': 'nl', 'TD': 'bl', 'TI': 'bl', 'TKM': 'bl', 'TN': 'nl', 'TS': 'bl', 'TSM': 'bl', 'TV': 'bl', 
        'TX': 'nl', 'UDV': 'bl', 'UP': 'bl', 'UI': 'bl', 'VA': 'bl', 'ZS': 'bl', 'ZZ': 'bl'
    }

    label_map = {'nl':0, 'bl':1, 'bb':2}
    
    inverse_label_map = {0:'침엽수', 1:'활엽수', 2:'기타수종'}

    return label2kind, label_map, inverse_label_map
import torch.optim as optim

def get_optimizer(name, params, lr, weight_decay):
    if name == 'SGD':
        return optim.SGD(params=params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif name == 'Adam':
        return optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError('Only Adam and SGD are implemented!')

# ---------------------------------------------------------------------------------------------
# Modified from Swin Transformer, please refer to https://github.com/microsoft/Swin-Transformer
# ---------------------------------------------------------------------------------------------

from torch import optim as optim


def build_optimizer(config, parameters):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer
#
def build_optimizerv2(config, parameters):
    """
    config has .optimizer  Python Dict
    from mmseg.core import build_optimizer
    optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
    config.optimizer=optimizer
    optimizer = Omegaconf.to_object(config.optimizer)
    opt = build_optimizer(model,optimizer) 
    """
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer
#
#
# def set_weight_decay(model, skip_list=(), skip_keywords=()):
#     def check_keywords_in_name(name, keywords=()):
#         isin = False
#         for keyword in keywords:
#             if keyword in name:
#                 isin = True
#         return isin
#
#     has_decay = []
#     no_decay = []
#
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue  # frozen weights
#         if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
#                 check_keywords_in_name(name, skip_keywords):
#             no_decay.append(param)
#             # print(f"{name} has no weight decay")
#         else:
#             has_decay.append(param)
#     return [{'params': has_decay},
#             {'params': no_decay, 'weight_decay': 0.}]
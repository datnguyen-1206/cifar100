import torch 
from .sam import SAM


def get_optimizer(model, opt_name='SGD', opt_hyperparameter={}):
    if (opt_name=='SGD'):
        optimizer = torch.optim.SGD(model.parameters(), **opt_hyperparameter)
        return optimizer
    elif (opt_name=='SAM'):
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, **opt_hyperparameter)
        return optimizer
    else :
        raise ValueError(f'Optimizer {opt_name} not supported')
    
def get_scheduler(
    optimizer, 
    sch_name='cosine',
    sch_hyperparameter={}
):
    if sch_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        **sch_hyperparameter
    )
    elif sch_name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(
        optimizer,
        **sch_hyperparameter
    )
    elif sch_name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        **sch_hyperparameter
    )
    else:
        raise ValueError("Invalid scheduler!!!")
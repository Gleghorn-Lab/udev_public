import torch.nn as nn


def get_loss_fct(task_type):
    if task_type == 'singlelabel' or task_type == 'tokenwise':
        loss_fct = nn.CrossEntropyLoss()
    elif task_type == 'multilabel':
        loss_fct = nn.BCEWithLogitsLoss()
    elif task_type == 'regression':
        loss_fct = nn.MSELoss()
    else:
        print(f'Specified wrong classification type {task_type}')
    return loss_fct

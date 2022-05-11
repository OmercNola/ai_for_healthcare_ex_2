from __future__ import absolute_import, division, print_function
import torch
from pathlib import Path

# save checkpoit:
def save_model_checkpoint(model, epoch):
    """
    """
    model_state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    torch.save(
        {'epoch': epoch,
         'model_state_dict': model_state_dict},
        Path(f'checkpoints/epoch_{epoch}_.pt'))

    print(f'checkpoint has been saved !')

# load checkpoint:
def load_model_checkpoint(path_, model):
    """
    """

    # load the checkpoint:
    checkpoint = torch.load(str(Path(path_)), map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint['epoch']

    return model, epoch
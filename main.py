# import libraries
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.image as image
from tqdm import tqdm
import pydicom
import sys
import os
from functools import partial
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
"====================================="
import torch
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
import torchvision.models as models
import torchvision.transforms as T
import albumentations as A
from torch.autograd import Variable
"====================================="
from dataset import MaskDataset
from utils import get_infor, Visualize_image
from glob import glob
from ipdb import set_trace
"================================"
from model import Unet
from PIL import Image
from utils import parallel_func, parallel_visualize, visualize_dataset
from losses import *
from dataset import toTensor, toNumpy
from torch import distributed as dist

def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0


def train(args, model, train_loader, optimizer):

    for epoch in range(100):

        number_iter = 1

        with tqdm(train_loader, unit="batch") as tepoch:

            for imgs, masks in tepoch:

                imgs = toTensor(imgs).float()
                masks = toTensor(masks).float()

                tepoch.set_description(f"Epoch {epoch}, interation {number_iter}")
                optimizer.zero_grad()

                imgs_gpu = imgs.to(args.device)
                outputs = model(imgs_gpu)
                masks = masks.to(args.device)

                dice_scores = dice_score(outputs, masks)
                loss = combo_loss(outputs, masks)

                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item(),
                                   dice_score=dice_scores.item())
                number_iter += 1


def main(args, init_distributed=False):

    """
    :param args:
    :type args:
    :param init_distributed:
    :type init_distributed:
    :return:
    :rtype:
    """
    "================================================================================="
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        torch.cuda.empty_cache()
        torch.cuda.init()
        args.device = torch.device("cuda", args.device_id)

    try:
        print(f'rank: {args.rank}')
        if init_distributed:
            dist.init_process_group(
                backend=args.backend,
                init_method=args.init_method,
                world_size=args.world_size,
                rank=args.rank,
                timeout=timedelta(seconds=120)
            )
    except Exception as e:
        print(e)

    is_distributed = args.world_size > 1
    print(f'rank: {args.rank} after  dist.init_process_group')
    "================================================================================="

    # create our model:
    # create the model:
    model_ft = models.resnet50(pretrained=True)
    model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
    model = Unet(model_ft)

    # create optimizer with trainable params:
    train_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(train_params, lr=0.001, betas=(0.9, 0.99))
    "================================================================================="

    # model to device:
    model.to(args.device)

    print(f'rank: {args.rank} after model init')
    "================================================================================="
    """Parallel"""
    if torch.cuda.is_available():
        # if we have more than 1 gpu:
        if args.world_size > 1:

            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            model = DDP(
                model,
                device_ids=[args.device_id],
                output_device=args.device_id,
                find_unused_parameters=True,
                broadcast_buffers=False
            )
            model.to(args.device)

            # Dataloaders:
            # read data:
            train_imgs = sorted(glob('raw_data/dicom-images-train/**/*.dcm', recursive=True))
            test_imgs = sorted(glob('raw_data/dicom-images-test/**/*.dcm', recursive=True))
            print(f'Number of train files: {len(train_imgs)}')
            print(f'Number of test files : {len(test_imgs)}')
            train_df = pd.read_csv('raw_data/train-rle.csv')

            print("Loading information for training set \n")
            # set_trace()
            parallel_func = partial(parallel_func, df=train_df, file_paths=train_imgs)
            train_infor = get_infor(train_df, parallel_func)
            print("information has been loaded ! \n")

            # Visualize image and mask:
            if is_master():
                Visualize_image(train_df, train_imgs)

            # create transforms:
            train_transform = A.Compose([
                A.HorizontalFlip(),
                A.OneOf([
                    A.RandomBrightnessContrast(),
                    A.RandomGamma(),
                ], p=0.3),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    A.GridDistortion(),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ], p=0.3),
                A.ShiftScaleRotate(),
            ])

            train_dataset = MaskDataset(train_df, train_infor, train_transform)

            sampler = DistributedSampler(
                dataset=train_dataset,
                num_replicas=2,
                rank=args.rank,
                shuffle=args.shuffle,
                seed=args.seed
            )

            # create train datatset and dataloader:

            train_loader = DataLoader(train_dataset,
                                      batch_size=30,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=4,
                                      sampler=sampler)

    "================================================================================="
    """Training"""

    train(args=args,
          model=model,
          optimizer=optimizer,
          train_loader=train_loader,
          train_sampler=train_sampler)

    "================================================================================="
    """cleanup"""
    print(f'rank: {args.rank} at the end, wating for cleanup')
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


def distributed_main(device_id, args):
    """
    :param device_id:
    :type device_id:
    :param args:
    :type args:
    :return:
    :rtype:
    """
    args.device_id = device_id
    if args.rank is None:
        args.rank = args.start_rank + device_id
        args.local_rank = args.rank
    main(args, init_distributed=True)


if __name__ == '__main__':

    "================================================================================="
    parser = argparse.ArgumentParser(description='ex2')
    "================================================================================="
    parser.add_argument('--device', type=torch.device,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='device type')
    "================================================================================="
    "Train settings 1"
    parser.add_argument('--save_model_during_training', action="store_true", default=False,
                        help='save model during training ? ')
    parser.add_argument('--save_model_every', type=int, default=600,
                        help='when to save the model - number of batches')
    parser.add_argument('--print_loss_every', type=int, default=25,
                        help='when to print the loss - number of batches')
    parser.add_argument('--print_eval_every', type=int, default=50,
                        help='when to print f1 scores during eval - number of batches')
    parser.add_argument('--checkpoint_path', type=str,
                        default=None,
                        help='checkpoint path for evaluation or proceed training ,'
                             'if set to None then ignor checkpoint')
    "================================================================================="
    "Hyper-parameters"
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')  # every 2 instances are using 1 "3090 GPU"
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='learning rate (default: 0.00001) took from longformer paper')
    parser.add_argument('--dropout_p', type=float, default=0.25,
                        help='dropout_p (default: 0.1)')
    parser.add_argument('--use_scheduler', type=bool, default=False,
                        help='use linear scheduler with warmup ?')
    parser.add_argument('--num_warmup_steps', type=int, default=50,
                        help='number of warmup steps in the scheduler, '
                             '(just if args.use_scheduler is True)')
    parser.add_argument('--beta_1', type=float, default=0.9,
                        help='beta 1 for AdamW. default=0.9')
    parser.add_argument('--beta_2', type=float, default=0.999,
                        help='beta 2 for AdamW. default=0.999')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight_decay for AdamW. default=0.0001')
    parser.add_argument('--use_clip_grad_norm', type=bool, default=False,
                        help='clip grad norm to args.max_grad_norm')
    parser.add_argument('--max_grad_norm', type=float, default=40,
                        help='max norm for gradients cliping '
                             '(just if args.use_clip_grad_norm is True)')
    parser.add_argument('--sync-bn', action='store_true', default=True,
                        help='sync batchnorm')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='shuffle')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='number of workers in dataloader')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='prefetch factor in dataloader')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    "================================================================================="
    args = parser.parse_known_args()[0]



    # print 10 images from dataset:
    visualize_dataset(train_dataset, parallel_visualize)

    # val_dataset = MaskDataset(val_df, val_infor)
    # val_loader = DataLoader(val_dataset, batch_size=20, shuffle=True, drop_last=True)



    # freeze some params:
    for i, child in enumerate(model.children()):
        if i <= 7:
            for param in child.parameters():
                param.requires_grad = False

    # model summary:
    print(summary(model,input_size=(1,512,512)))



    # train the model:
    train(args, model, train_loader, optimizer)
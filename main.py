import time
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.image as image
from tqdm import tqdm
import pydicom
from datetime import datetime, timedelta
import sys
import os
import random
import platform
from functools import partial
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from sampler import DistributedEvalSampler
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
import torchvision.models as models
import torchvision.transforms as T
import albumentations as A
from torch.autograd import Variable
from dataset import MaskDataset
from utils import get_infor, Visualize_image, plot_image_during_training
from glob import glob
import copy
from ipdb import set_trace
from model import Unet
from PIL import Image
from losses import *
from dataset import toTensor, toNumpy
from torch import distributed as dist
from saver_and_loader import save_model_checkpoint, load_model_checkpoint

def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0


def train(args, model, train_loader, test_loader, optimizer, train_sampler):

    model.train()

    is_distributed = args.world_size > 1

    for epoch in range(args.epochs):

        number_iter = 1

        if is_master():
            print(f'training... epoch {epoch}')

        if is_distributed:
            train_sampler.set_epoch(epoch)
            dist.barrier()

        with tqdm(train_loader, unit="batch") as tepoch:

            for batch_counter, (imgs, masks) in enumerate(tepoch):

                signal = torch.tensor([1], device=args.device)
                work = dist.all_reduce(signal, async_op=True)
                work.wait()
                if signal.item() < args.world_size:
                    continue
                "=================================================================="

                imgs = toTensor(imgs).float()
                masks = toTensor(masks).float()

                tepoch.set_description(f"Epoch {epoch}, interation {number_iter}")
                optimizer.zero_grad()

                imgs_gpu = imgs.to(args.device)
                outputs = model(imgs_gpu)
                masks = masks.to(args.device)

                if is_master() and (((batch_counter + 1) % args.save_model_every) == 0):
                    if (epoch > 80) and ((epoch % 2) == 0):
                        if args.save_model_during_training:
                            save_model_checkpoint(model, epoch)

                dice_scores = dice_score(outputs, masks)
                loss = combo_loss(outputs, masks)

                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item(),
                                   dice_score=dice_scores.item())
                number_iter += 1

        if signal.item() >= args.world_size:
            dist.all_reduce(torch.tensor([0], device=args.device))

        dist.barrier()

        if is_master():
            for batch_counter, (imgs, masks) in enumerate(test_loader):

                if batch_counter == 2:
                    break

                imgs = toTensor(imgs).float()
                masks = toTensor(masks).float()

                imgs_gpu = imgs.to(args.device)
                masks = masks.to(args.device)

                with torch.no_grad():
                    model_ = copy.deepcopy(model.module)
                    model_.eval()
                    outputs = model_(imgs_gpu)

                plot_image_during_training(outputs, masks, imgs_gpu)


def main(args, init_distributed=False):

    """
    :param args:
    :type args:
    :param init_distributed:
    :type init_distributed:
    :return:
    :rtype:
    """

    from utils import parallel_func, parallel_visualize, visualize_dataset
    "================================================================================="
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        torch.cuda.empty_cache()
        torch.cuda.init()
        args.device = torch.device("cuda", args.device_id)

    try:
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
    "================================================================================="

    # create our model:
    model_ft = models.resnet50(pretrained=True)
    model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
    model = Unet(model_ft)

    path_ = args.checkpoint_path
    model, _ = load_model_checkpoint(path_, model)

    # freeze some params:
    for i, child in enumerate(model.children()):
        if i <= 7:
            for param in child.parameters():
                param.requires_grad = False

    # model summary:
    if is_master():
        model.to(args.device)
        print(summary(model, input_size=(1, 512, 512)))


    # create optimizer with trainable params:
    train_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(train_params, lr=0.001, betas=(0.9, 0.99))
    "================================================================================="

    # model to device:
    model.to(args.device)

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
                find_unused_parameters=False,
                broadcast_buffers=False
            )
            model.to(args.device)

            # Dataloaders:
            # read data:
            train_imgs = sorted(glob('raw_data/dicom-images-train/**/*.dcm', recursive=True))
            test_imgs = sorted(glob('raw_data/dicom-images-test/**/*.dcm', recursive=True))
            if is_master():
                print(f'Number of train files: {len(train_imgs)}')
                print(f'Number of test files : {len(test_imgs)}')

            # read labled data
            train_df = pd.read_csv('raw_data/train-rle.csv')

            if is_master():
                print("Loading information for training set \n")
            parallel_func = partial(parallel_func, df=train_df, file_paths=train_imgs)
            infor = get_infor(train_df, parallel_func)

            random.shuffle(infor)

            train_infor = infor[:int(len(infor) * 0.8)]
            test_infor = infor[int(len(infor) * 0.8):]

            if is_master():
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
            test_dataset = MaskDataset(train_df, test_infor)

            # print 10 images from dataset:
            if is_master():
                visualize_dataset(train_dataset, parallel_visualize)

            train_sampler = DistributedSampler(
                dataset=train_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=args.shuffle,
                seed=args.seed
            )

            # create dataloader:
            train_loader = DataLoader(train_dataset,
                                      batch_size=30,
                                      shuffle=False,
                                      drop_last=True,
                                      num_workers=args.num_workers,
                                      persistent_workers=True,
                                      prefetch_factor=args.prefetch_factor,
                                      sampler=train_sampler,
                                      pin_memory=True)


            test_loader = DataLoader(test_dataset,
                                     shuffle=True,
                                     drop_last=True,
                                     batch_size=args.single_rank_batch_size,
                                     prefetch_factor=args.prefetch_factor,
                                     num_workers=args.num_workers,
                                     persistent_workers=True,
                                     pin_memory=True
            )
    "================================================================================="
    """Training"""
    train(args=args,
          model=model,
          optimizer=optimizer,
          train_loader=train_loader,
          test_loader=test_loader,
          train_sampler=train_sampler)
    "================================================================================="
    """cleanup"""
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
    __file__ = "main.py"
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
    parser.add_argument('--save_model_every', type=int, default=159,
                        help='when to save the model - number of batches')
    parser.add_argument('--print_loss_every', type=int, default=25,
                        help='when to print the loss - number of batches')
    parser.add_argument('--plot_images_during_training', type=int, default=5,
                        help='when to plot image - number of batches')
    parser.add_argument('--print_eval_every', type=int, default=50,
                        help='when to print f1 scores during eval - number of batches')
    parser.add_argument('--checkpoint_path', type=str,
                        default='checkpoints/epoch_148_.pt',
                        help='checkpoint path for evaluation or proceed training ,'
                             'if set to None then ignor checkpoint')
    "================================================================================="
    parser.add_argument('--world_size', type=int, default=2,
                        help='if None - will be number of devices')
    parser.add_argument('--start_rank', default=0, type=int,
                        help='we need to pass diff values if we are using multiple machines')
    parser.add_argument("--local_rank", type=int)
    "================================================================================="
    "Hyper-parameters"
    parser.add_argument('--epochs', type=int, default=150,
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
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers in dataloader')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='prefetch factor in dataloader')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    "================================================================================="
    args = parser.parse_known_args()[0]

    print(f'Available devices: {torch.cuda.device_count()}\n')
    "================================================================================="
    # Ensure deterministic behavior
    torch.use_deterministic_algorithms(True)
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    "================================================================================="
    """Distributed:"""

    # multiple nodes:
    # args.world_size = args.gpus * args.nodes

    # single node:
    if (args.world_size is None):
        args.world_size = torch.cuda.device_count()

    # check platform:
    IsWindows = platform.platform().startswith('Win')

    # if single GPU:
    if args.world_size == 1:
        # on nvidia 3090:
        args.batch_size = 2
        args.single_rank_batch_size = 2
        args.device_id = 0
        args.rank = 0
        main(args)

    # DDP for multiple GPU'S:
    elif args.world_size > 1:

        print(f'world_size: {args.world_size}')

        args.local_world_size = torch.cuda.device_count()

        # for nvidia 3090 or titan rtx (24GB each)
        args.batch_size = args.local_world_size * 2

        args.single_rank_batch_size = int(args.batch_size / args.local_world_size)

        port = random.randint(10000, 20000)
        args.init_method = f'tcp://127.0.0.1:{port}'
        # args.init_method = f'tcp://192.168.1.101:{port}'
        # args.init_method = 'env://'

        # we will set the rank in distributed main function
        args.rank = None

        # 'nccl' is the fastest, but doesnt woek in windows.
        args.backend = 'gloo' if IsWindows else 'nccl'

        # open args.local_world_size new process in each node:
        mp.spawn(fn=distributed_main, args=(args,), nprocs=args.local_world_size, )

    else:
        args.device = torch.device("cpu")
        args.single_rank_batch_size = args.batch_size
        main(args)
    "================================================================================="


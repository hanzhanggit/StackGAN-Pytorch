from __future__ import print_function
import torch
import torchvision.transforms as transforms

import argparse
import datetime
import dateutil.tz
import os
import pprint
import random
import sys

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.datasets import TextDataset
from miscc.config import cfg, cfg_from_file
from trainer import GANTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='birds_stage1.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.data_dir:
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)
    if not args.manualSeed:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    num_gpu = len(cfg.GPU_ID.split(','))
    if cfg.TRAIN.FLAG:
        image_transform = transforms.Compose([
            transforms.RandomCrop(cfg.IMSIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                              imsize=cfg.IMSIZE,
                              transform=image_transform)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        algo = GANTrainer(output_dir)
        algo.train(dataloader, cfg.STAGE)
    else:
        datapath = '%s/test/val_captions.t7' % (cfg.DATA_DIR)
        algo = GANTrainer(output_dir)
        algo.sample(datapath, cfg.STAGE)

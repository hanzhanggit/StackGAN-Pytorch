from __future__ import print_function
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
from torch.utils.data import DataLoader

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.datasets import TextDataset
from miscc.config import cfg, cfg_from_file
from trainer import GANTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/coco_s1.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--STAGE1_G', dest='STAGE1_G', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed', default=47)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    print(cfg)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    if args.STAGE1_G != '':
        cfg.STAGE1_G = args.STAGE1_G
    print('Using config:')
    pprint.pprint(cfg)
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = 'output/%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    print("Output:", output_dir)
    if cfg.STAGE == 1:
        with open("train_stage2.sh", "w") as fp:
            fp.write("python code/main.py --cfg {} --STAGE1_G {}\n".format(
                args.cfg_file.replace("s1", "s2"),
                os.path.join(output_dir, "Model", "netG_epoch_{}.pth".format(cfg.TRAIN.MAX_EPOCH - 1))
            ))

    num_gpu = len(cfg.GPU_ID.split(','))
    if cfg.TRAIN.FLAG:
        # prepare image transforms
        image_transform = transforms.Compose([
            transforms.RandomCrop(cfg.IMSIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # prepare Text caption
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                              imsize=cfg.IMSIZE,
                              embedding_type=cfg.EMBEDDING_TYPE,
                              transform=image_transform)
        print("Dataset Length:", len(dataset))
        assert dataset
        dataloader = DataLoader(dataset=dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
                                drop_last=cfg.TRAIN.BATCH_DROP_LAST,
                                shuffle=True, num_workers=int(cfg.WORKERS))

        algo = GANTrainer(output_dir)
        algo.train(dataloader, cfg.STAGE)
    else:
        datapath = '%s/test/val_captions.t7' % cfg.DATA_DIR
        algo = GANTrainer(output_dir)
        algo.sample(datapath, cfg.STAGE)

from __future__ import print_function

import pathlib
import shutil

import PIL
import numpy as np
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
from PIL.Image import Image
from git import Repo
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import _setup_size

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
    parser.add_argument('--test_phase', dest='test_phase', default=False, action='store_true')
    parser.add_argument('--NET_G', dest='NET_G', default='', help="Path to generator for testing")
    parser.add_argument('--NET_D', dest='NET_D', default='', help="Path to discriminator for testing")
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--STAGE1_G', dest='STAGE1_G', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed', default=47)
    args = parser.parse_args()
    return args


class AspectResize(torch.nn.Module):
    """
   Resize image while keeping the aspect ratio.
   Extra parts will be covered with 255(white) color value
   """
    
    def __init__(self, size, background=255):
        super().__init__()
        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.background = background
    
    @staticmethod
    def fit_image_to_canvas(image: Image, canvas_width, canvas_height, background=255) -> Image:
        # Get the dimensions of the image
        image_width, image_height = image.size
        
        # Calculate the aspect ratio of the image
        image_aspect_ratio = image_width / float(image_height)
        
        # Calculate the aspect ratio of the canvas
        canvas_aspect_ratio = canvas_width / float(canvas_height)
        
        # Calculate the new dimensions of the image to fit the canvas
        if canvas_aspect_ratio > image_aspect_ratio:
            new_width = canvas_height * image_aspect_ratio
            new_height = canvas_height
        else:
            new_width = canvas_width
            new_height = canvas_width / image_aspect_ratio
        
        # Resize the image to the new dimensions
        image = image.resize((int(new_width), int(new_height)), PIL.Image.BICUBIC)
        
        # Create a blank canvas of the specified size
        canvas = np.zeros((int(canvas_height), int(canvas_width), 3), dtype=np.uint8)
        canvas[:, :, :] = background
        
        # Calculate the position to paste the resized image on the canvas
        x = int((canvas_width - new_width) / 2)
        y = int((canvas_height - new_height) / 2)
        
        # Paste the resized image onto the canvas
        canvas[y:y + int(new_height), x:x + int(new_width)] = np.array(image)
        
        return PIL.Image.fromarray(canvas)
    
    def forward(self, image: Image) -> Image:
        image = self.fit_image_to_canvas(image, self.size[0], self.size[1], self.background)
        return image


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    if args.STAGE1_G != '':
        cfg.STAGE1_G = args.STAGE1_G
    if args.test_phase:
        cfg.TRAIN.FLAG = False
        if args.NET_G:
            cfg.TRAIN.FINETUNE.FLAG = True
            cfg.TRAIN.FINETUNE.NET_G = args.NET_G
            cfg.TRAIN.FINETUNE.NET_D = args.NET_D
    print('Using config:')
    pprint.pprint(cfg)
    pprint.pprint(args)
    # save git checksum
    project_root = pathlib.Path(__file__).parents[0]
    repo = Repo(project_root)
    args.git_checksum = repo.git.rev_parse("HEAD")  # save commit checksum
    
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    phase = "test" if args.test_phase else "train"
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = 'output/%s_%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, phase, timestamp)
    
    if cfg.STAGE == 1:
        # STAGE-1
        if cfg.TRAIN.FLAG:
            # STAGE-1 TRAINING
            # prepare script for stage-2 training
            with open("train_stage2.sh", "w") as fp:
                fp.write(
                    "#!/usr/bin/bash\nsh code/miscc/cuda_mem.sh\n"
                    "python code/main.py --cfg {} --manualSeed 47 --STAGE1_G {}\n".format(
                        args.cfg_file.replace("s1", "s2"),
                        os.path.join(output_dir, "Model", "netG_epoch_{}.pth".format(cfg.TRAIN.MAX_EPOCH - 1))
                    ))
            # prepare script for stage-1 testing
            with open("test_stage1.sh", "w") as fp:
                fp.write(
                    "#!/usr/bin/bash\nsh code/miscc/cuda_mem.sh\n"
                    "python code/main.py --test_phase --manualSeed 47 --cfg {} --NET_G {} --NET_D {}\n".format(
                        args.cfg_file,
                        os.path.join(output_dir, "Model", "netG_epoch_{}.pth".format(cfg.TRAIN.MAX_EPOCH)),
                        os.path.join(output_dir, "Model", "netD_epoch_last.pth"),
                    ))
        else:
            # STAGE-1 TESTING
            ...
    else:
        # STAGE-2
        if cfg.TRAIN.FLAG:
            # STAGE-2 TRAINING
            # prepare script for stage-2 testing
            with open("test_stage2.sh", "w") as fp:
                fp.write(
                    "#!/usr/bin/bash\nsh code/miscc/cuda_mem.sh\n"
                    "python code/main.py --test_phase --manualSeed 47 --cfg {} --NET_G {} --NET_D {}\n".format(
                        args.cfg_file,
                        os.path.join(output_dir, "Model", "netG_epoch_{}.pth".format(cfg.TRAIN.MAX_EPOCH)),
                        os.path.join(output_dir, "Model", "netD_epoch_last.pth"),
                    ))
        else:
            # STAGE-2 TESTING
            ...
    
    num_gpu = len(cfg.GPU_ID.split(','))
    if cfg.TRAIN.FLAG:
        # prepare image transforms
        image_transform = transforms.Compose([
            transforms.RandomCrop(cfg.IMSIZE) if False else AspectResize(cfg.IMSIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # prepare Text caption
        train_dataset = TextDataset(cfg.DATA_DIR, 'train',
                                    imsize=cfg.IMSIZE,
                                    embedding_type=cfg.EMBEDDING_TYPE,
                                    transform=image_transform,
                                    float_precision=32)
        # prepare Text caption
        test_dataset = TextDataset(cfg.DATA_DIR, 'test',
                                   imsize=cfg.IMSIZE,
                                   embedding_type=cfg.EMBEDDING_TYPE,
                                   transform=image_transform,
                                   float_precision=32)
        print("Dataset Length:", len(train_dataset))
        assert train_dataset
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
                                      drop_last=cfg.TRAIN.BATCH_DROP_LAST,
                                      shuffle=True, num_workers=int(cfg.WORKERS))
        
        algo = GANTrainer(output_dir)
        shutil.copyfile(args.cfg_file, os.path.join(output_dir, os.path.basename(args.cfg_file)))
        with open(os.path.join(output_dir, "config.txt"), "w") as fp:
            fp.write("%s\n" % (str(args)))
            fp.write("%s" % (str(cfg)))
        algo.train(train_dataloader, cfg.STAGE, test_dataset)
    else:
        datapath = os.path.join(cfg.DATA_DIR, "test", cfg.EMBEDDING_TYPE)
        if os.path.isfile(datapath):
            algo = GANTrainer(output_dir)
            shutil.copyfile(args.cfg_file, os.path.join(output_dir, os.path.basename(args.cfg_file)))
            algo.sample(datapath, output_dir, cfg.STAGE)

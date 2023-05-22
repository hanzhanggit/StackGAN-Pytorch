from __future__ import print_function

from pprint import pprint

from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import gc

import numpy as np
import torchfile
from torch.utils.tensorboard import SummaryWriter

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init
from miscc.utils import save_img_results, save_model
from miscc.utils import KL_loss
from miscc.utils import compute_discriminator_loss, compute_generator_loss

from tensorboard import summary
from tensorboardX import FileWriter


class GANTrainer(object):
    def __init__(self, output_dir):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = SummaryWriter(self.log_dir)

        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

    # ############# For training stageI GAN #############
    def load_network_stageI(self):
        from model import STAGE1_G, STAGE1_D
        netG = STAGE1_G()
        netG.apply(weights_init)
        print(netG)
        netD = STAGE1_D()
        netD.apply(weights_init)
        print(netD)
        if cfg.TRAIN.FINETUNE.FLAG:
            assert os.path.isfile(
                cfg.TRAIN.FINETUNE.NET_G), "TRAIN.FINETUNE.NET_G is required when TRAIN.FINETUNE.FLAG=True"
            assert os.path.isfile(
                cfg.TRAIN.FINETUNE.NET_D), "TRAIN.FINETUNE.NET_D is required when TRAIN.FINETUNE.FLAG=True"

            state_dict = torch.load(cfg.TRAIN.FINETUNE.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from NET_G: ', cfg.TRAIN.FINETUNE.NET_G)

            state_dict = torch.load(cfg.TRAIN.FINETUNE.NET_D, map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from NET_D: ', cfg.TRAIN.FINETUNE.NET_D)
        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    # ############# For training stageII GAN  #############
    def load_network_stageII(self):
        from model import STAGE1_G, STAGE2_G, STAGE2_D

        Stage1_G = STAGE1_G()
        netG = STAGE2_G(Stage1_G)
        netG.apply(weights_init)
        netD = STAGE2_D()
        netD.apply(weights_init)
        print(netG)
        print(netD)
        if cfg.TRAIN.FINETUNE.FLAG:
            assert os.path.isfile(
                cfg.TRAIN.FINETUNE.NET_G), "TRAIN.FINETUNE.NET_G is required when TRAIN.FINETUNE.FLAG=True"
            assert os.path.isfile(
                cfg.TRAIN.FINETUNE.NET_D), "TRAIN.FINETUNE.NET_D is required when TRAIN.FINETUNE.FLAG=True"

            state_dict = torch.load(cfg.TRAIN.FINETUNE.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from NET_G: ', cfg.TRAIN.FINETUNE.NET_G)

            state_dict = torch.load(cfg.TRAIN.FINETUNE.NET_D, map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from NET_D: ', cfg.TRAIN.FINETUNE.NET_D)

        if cfg.STAGE1_G != '' and os.path.isfile(cfg.STAGE1_G):
            state_dict = torch.load(cfg.STAGE1_G, map_location=lambda storage, loc: storage)
            netG.STAGE1_G.load_state_dict(state_dict)
            print('Load from STAGE1_G: ', cfg.STAGE1_G)
        else:
            assert ValueError("Please give the STAGE1_G path while training Stage-2 of StackGAN")
        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    def train(self, data_loader, stage=1):
        if stage == 1:
            netG, netD = self.load_network_stageI()
        else:
            netG, netD = self.load_network_stageII()

        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, nz))
        with torch.no_grad():
            fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()

        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
        optimizerD = optim.Adam(netD.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para, lr=cfg.TRAIN.GENERATOR_LR, betas=(0.5, 0.999))
        # setup epoch
        epoch_start = 0
        if cfg.TRAIN.FINETUNE.FLAG:
            epoch_start = cfg.TRAIN.FINETUNE.EPOCH_START

        count = 0
        for epoch in range(epoch_start, self.max_epoch):
            start_t = time.time()
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr

            loop_ran = False
            for i, data in enumerate(data_loader, 0):
                loop_ran = True
                ######################################################
                # (1) Prepare training data
                ######################################################
                real_img_cpu, txt_embedding = data
                real_imgs = Variable(real_img_cpu)
                txt_embedding = Variable(txt_embedding)
                if cfg.CUDA:
                    real_imgs = real_imgs.cuda()
                    txt_embedding = txt_embedding.cuda()

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                inputs = (txt_embedding, noise)
                _, fake_imgs, mu, logvar = nn.parallel.data_parallel(netG, inputs, self.gpus)

                ############################
                # (3) Update D network
                ###########################
                netD.zero_grad()
                errD, errD_real, errD_wrong, errD_fake = compute_discriminator_loss(netD, real_imgs, fake_imgs,
                                                                                    real_labels, fake_labels,
                                                                                    mu, self.gpus)
                errD.backward()
                optimizerD.step()
                ############################
                # (2) Update G network
                ###########################
                netG.zero_grad()
                errG = compute_generator_loss(netD, fake_imgs,
                                              real_labels, mu, self.gpus)
                kl_loss = KL_loss(mu, logvar)
                errG_total = errG + kl_loss * cfg.TRAIN.COEFF.KL
                errG_total.backward()
                optimizerG.step()

                count = count + 1
                if i % 100 == 0:
                    self.summary_writer.add_scalar('D_loss', errD.data, count)
                    self.summary_writer.add_scalar('D_loss_real', errD_real, count)
                    self.summary_writer.add_scalar('D_loss_wrong', errD_wrong, count)
                    self.summary_writer.add_scalar('D_loss_fake', errD_fake, count)
                    self.summary_writer.add_scalar('G_loss', errG.data, count)
                    self.summary_writer.add_scalar('KL_loss', kl_loss.data, count)
                if epoch % self.snapshot_interval == 0 and i % 100 == 0:
                    # save the image result for each epoch
                    inputs = (txt_embedding, fixed_noise)
                    lr_fake, fake, _, _ = nn.parallel.data_parallel(netG, inputs, self.gpus)
                    save_img_results(real_img_cpu, fake, epoch, self.image_dir)
                    if lr_fake is not None:
                        save_img_results(None, lr_fake, epoch, self.image_dir)
            if loop_ran is False:
                raise Warning(
                    "Not enough data available.\n"
                    "Reasons:\n"
                    "(1) Dataset() length=0 or \n"
                    "(2) When `drop_last=True` in Dataloader() and the `Dataset() length` < `batch-size`\n"
                    "Solutions:\n"
                    "(1) Reduce batch size to satisfy `Dataset() length` >= `batch-size`[recommended]\n"
                    "(2) Set `drop_last=False`[not recommended]")
            end_t = time.time()
            print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
                     Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                     Total Time: %.2fsec
                  '''
                  % (epoch, self.max_epoch, i, len(data_loader),
                     errD.data.item(), errG.data.item(), kl_loss.data.item(),
                     errD_real, errD_wrong, errD_fake, (end_t - start_t)))
            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD, epoch, self.model_dir)

            # CLEAN GPU RAM  ########################
            # pprint(torch.cuda.memory_summary(device=None, abbreviated=False))
            torch.cuda.empty_cache()
            del real_imgs
            del txt_embedding
            del inputs
            del _
            del fake_imgs
            del mu
            del logvar
            del errD
            del errD_real
            del errD_wrong
            del errD_fake
            del kl_loss
            del errG_total
            gc.collect()
            # CLEAN GPU RAM ########################
        #
        save_model(netG, netD, self.max_epoch, self.model_dir)
        #
        self.summary_writer.flush()
        self.summary_writer.close()

    def sample(self, datapath, stage=1):
        if stage == 1:
            netG, _ = self.load_network_stageI()
        else:
            netG, _ = self.load_network_stageII()
        netG.eval()

        # Load text embeddings generated from the encoder
        t_file = torchfile.load(datapath)
        captions_list = t_file.raw_txt
        embeddings = np.concatenate(t_file.fea_txt, axis=0)
        num_embeddings = len(captions_list)
        print('Successfully load sentences from: ', datapath)
        print('Total number of sentences:', num_embeddings)
        print('num_embeddings:', num_embeddings, embeddings.shape)
        # path to save generated samples
        save_dir = cfg.NET_G[:cfg.NET_G.find('.pth')]
        mkdir_p(save_dir)

        batch_size = np.minimum(num_embeddings, self.batch_size)
        nz = cfg.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        if cfg.CUDA:
            noise = noise.cuda()
        count = 0
        while count < num_embeddings:
            if count > 3000:
                break
            iend = count + batch_size
            if iend > num_embeddings:
                iend = num_embeddings
                count = num_embeddings - batch_size
            embeddings_batch = embeddings[count:iend]
            # captions_batch = captions_list[count:iend]
            txt_embedding = Variable(torch.FloatTensor(embeddings_batch))
            if cfg.CUDA:
                txt_embedding = txt_embedding.cuda()

            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            inputs = (txt_embedding, noise)
            _, fake_imgs, mu, logvar = \
                nn.parallel.data_parallel(netG, inputs, self.gpus)
            for i in range(batch_size):
                save_name = '%s/%d.png' % (save_dir, count + i)
                im = fake_imgs[i].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                # print('im', im.shape)
                im = np.transpose(im, (1, 2, 0))
                # print('im', im.shape)
                im = Image.fromarray(im)
                im.save(save_name)
            count += batch_size

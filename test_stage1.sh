#!/usr/bin/bash
sh code/miscc/cuda_mem.sh
python code/main.py --test_phase --manualSeed 47 --cfg code/cfg/sixray_500_s1.yml --NET_G output/sixray_2381_ftt_1024D_cbow_nocrop_stage_1_train_2023_06_08_18_48_49\Model\netG_epoch_1000.pth --NET_D output/sixray_2381_ftt_1024D_cbow_nocrop_stage_1_train_2023_06_08_18_48_49\Model\netD_epoch_last.pth

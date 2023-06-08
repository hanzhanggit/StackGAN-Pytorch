#!/usr/bin/bash
sh code/miscc/cuda_mem.sh
python code/main.py --cfg code/cfg/sixray_500_s2.yml --manualSeed 47 --STAGE1_G output/sixray_2381_ftt_1024D_cbow_nocrop_stage_1_train_2023_06_08_18_48_49\Model\netG_epoch_999.pth

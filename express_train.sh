#!/usr/bin/bash
# REMOTE UBUNTU
# sh express_train.sh data/sixray_500  /data/fasttext/cc.en.300.bin  /data/Sixray_easy  /data/sixray_caption_db/<tip_gai.db>
# python data/generate_custom_dataset.py --data_dir data/sixray_500 --clean --copy_images --dataroot /data/Sixray_easy --fasttext_train_lr 0.05 --fasttext_train_algo skipgram --fasttext_train_epoch 50 --emb_dim 1000 --test_data_file data/sixray_500/test/test_captions.txt --sqlite /data/sixray_caption_db/<tip_gai.db>

# LOCAL UBUNTU
# sh express_train.sh "data/sixray_500" /mnt/c/Users/dndlssardar/Downloads/Fasttext/cc.en.300.bin /mnt/c/Users/dndlssardar/OneDrive\ -\ Smiths\ Group/Documents/Projects/Dataset/Sixray_easy/ data/tip_gai_20230525_1036.db

# python data/generate_custom_dataset.py --data_dir data/sixray_500 --sqlite data/tip_gai_20230525_1036.db --clean --copy_images --dataroot /mnt/c/Users/dndlssardar/OneDrive\ -\ Smiths Group/Documents/Projects/Dataset/Sixray_easy --fasttext_train_lr 0.01 --fasttext_train_algo skipgram --fasttext_train_epoch 50 --emb_dim 300 --test_data_file data/sixray_500/test/test_captions.txt
# Argument 1 must be a path to an sqlite database
export DATA_DIR="$1"  # data/sixray_500
export LM="$2"        # /data/fasttext/cc.en.300.bin
export DATA_ROOT="$3" # /data/Sixray_easy
export SQLITE_DB="$4" # /data/sixray_caption_db/tip_gai_20230523_1704.db
echo "--data_dir $DATA_DIR --fasttext_model $LM --dataroot $DATA_ROOT --sqlite $SQLITE_DB"

echo "############## CREATING DATASET  ##################"
# python data/generate_custom_dataset.py --data_dir "$DATA_DIR" --fasttext_model "$LM" --clean --copy_images --dataroot "$DATA_ROOT" --sqlite "$SQLITE_DB" --emb_dim 300
echo "############## TRAINING STAGE -1 ##################"
sh train_stage1.sh
echo "############## TRAINING STAGE -2 ##################"
sh train_stage2.sh

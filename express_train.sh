#!/usr/bin/bash
# sh express_train.sh "data/sixray_500"  "/data/fasttext/cc.en.300.bin"  "/data/Sixray_easy"  /data/sixray_caption_db/<tip_gai.db>
# Argument 1 must be a path to an sqlite database
export DATA_DIR="$1"  # data/sixray_500
export LM="$2"        # /data/fasttext/cc.en.300.bin
export DATA_ROOT="$3" # /data/Sixray_easy
export SQLITE_DB="$4" # /data/sixray_caption_db/tip_gai_20230523_1704.db
echo "--data_dir $DATA_DIR --fasttext_model $LM --dataroot $DATA_ROOT --sqlite $SQLITE_DB"

echo "############## CREATING DATASET  ##################"
python data/generate_custom_dataset.py --data_dir "$DATA_DIR" --fasttext_model "$LM" --clean --copy_images --dataroot "$DATA_ROOT" --sqlite "$SQLITE_DB" --emb_dim 300
echo "############## TRAINING STAGE -1 ##################"
sh train_stage1.sh
echo "############## TRAINING STAGE -2 ##################"
sh train_stage2.sh

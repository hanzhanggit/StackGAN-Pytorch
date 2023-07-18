#!/usr/bin/bash
exit()
# generate data form directory using pretrained fasttext model
python data/generate_custom_dataset.py --data_dir data/sixray_sample --fasttext_model /data/fasttext/cc.en.300.bin

# generate data form SQLite using pretrained fasttext model
python data/generate_custom_dataset.py --data_dir data/sixray_500 --fasttext_model /data/fasttext/cc.en.300.bin --clean --copy_images --dataroot /data/Sixray_easy --sqlite /data/sixray_caption_db/<tip_gai.db>

# generate data form SQLite while fasttext will be trained on out captions
python data/generate_custom_dataset.py --data_dir data/sixray_500 --clean --copy_images --dataroot /data/Sixray_easy/ --fasttext_train_lr 0.01 --fasttext_train_algo skipgram --fasttext_train_epoch 50 --emb_dim 300 --sqlite /data/sixray_caption_db/<tip_gai.db>
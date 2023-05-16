# Custom Data

## Prepare Directory

1. Save images into `data/my_data/train/images` directory
2. `data/my_data/train` should contain at-least two folders `images`  and `text`
3. The `text` directory contains a text file per image available in `images` directory

## Download Fasttext Model

4. `wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz` and save it.
5. `gunzip cc.en.300.bin.gz` will produce `cc.en.300.bin`
> **Note:** We can train our own Language Model. Documentation w.i.p


## Create Dataset
6. `python data/generate_custom_dataset.py --data_dir data/my_data --emb_dim 300 --fasttext_model /Fasttext/cc.en.300.bin`


import argparse
import os
import pathlib
from pprint import pprint

from tqdm import tqdm
from voc_tools.constants import VOC_IMAGES
from voc_tools.reader import list_dir
from voc_tools.utils import Dataset, VOCDataset

from dataset_wrap import SQLiteDataWrap, DatasetWrap
from langchain_openai_tools import OpenAITextLoader, OpenAICredentialManager, OpenAITextEmbeddingDB, OpenAIModelProxy
from easydict import EasyDict as edict


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--data_dir', dest='data_dir', help='Path to dataset directory.',
                        default='my_data', type=str, )
    parser.add_argument('--test_data_file', dest='test_data_file', help='A text file contains unseen captions to test',
                        default='', type=str, )
    parser.add_argument('--bulk', dest='bulk', default=False, action="store_true")
    parser.add_argument('--class_id', dest='class_id', default=False, action="store_true")
    parser.add_argument('--fasttext_train_lr', dest='fasttext_train_lr', type=float, default=None)
    parser.add_argument('--fasttext_train_algo', dest='fasttext_train_algo', type=str, default=None,
                        choices=['skipgram', 'cbow'])
    parser.add_argument('--fasttext_train_epoch', dest='fasttext_train_epoch', type=int, default=None)
    parser.add_argument('--fasttext_model', dest='fasttext_model', type=str, default=None)
    parser.add_argument('--emb_dim', dest='emb_dim', type=int, default=300)
    parser.add_argument('--openai_emb_db', dest='openai_emb_db', type=str, default=None)
    parser.add_argument('--sqlite', dest='sqlite',
                        help='Path to SQLite3 database file.',
                        default='', type=str, )
    parser.add_argument('--clean', dest='clean', default=False, action="store_true",
                        help="Clean before generating new data while using sqlite")
    parser.add_argument('--copy_images', dest='copy_images', default=False, action="store_true",
                        help="Copy images while generating new data using sqlite, --dataroot option is required")
    parser.add_argument('--dataroot', dest="dataroot", type=str,
                        help="This is a path to a image dataset to copy images while creating dataset using sqlite."
                             "The dataset format is default to PascalVOC format.")
    _args = parser.parse_args()
    pprint(_args)
    return _args


def generate_dataset(args):
    # creating an object to generate compatible pickle files for StackGAN
    # prepare fasttext training configuration
    fasttext_cfg = edict()
    fasttext_cfg.epoch = args.fasttext_train_epoch
    fasttext_cfg.lr = args.fasttext_train_lr
    fasttext_cfg.algorithm = args.fasttext_train_algo
    # read the test captions
    test_captions = None
    if os.path.isfile(args.test_data_file):
        with open(args.test_data_file) as fp:
            test_captions = fp.readlines()
    # load openai embedding DB
    if args.openai_emb_db:
        emb_model = OpenAIModelProxy(args.openai_emb_db)
        emb_name = "openai"
        args.emb_dim = emb_model.dim
    else:
        emb_model = None
        emb_name = None
    
    # initialize dataset
    vdw = DatasetWrap(args.data_dir, args.bulk, args.class_id,
                      fasttext_model_path=args.fasttext_model,
                      embedding_dimension=args.emb_dim,
                      test_captions=test_captions, emb_model=emb_model,
                      emb_model_name=emb_name)
    vdw.prepare_dataset(fasttext_cfg)


def create_openai_embedding_database(generate=False):
    from langchain.embeddings import OpenAIEmbeddings
    from openai.error import RateLimitError
    
    args = parse_args()
    
    embedding_database = args.openai_emb_db
    
    test_captions = []
    if os.path.isfile(args.test_data_file):
        with open(args.test_data_file) as fp:
            test_captions = fp.readlines()
    
    class Data:
        total_sentence = 0
        total_tokens = 0
    
    emb_db = OpenAITextEmbeddingDB(embedding_database)
    print("Items in DB: {}".format(len(emb_db.db)))
    
    # emb_db.query(DatasetWrap.clean("A quick brown fox jump over the lazy dog"))
    
    def caption_loader(additional_captions=()):
        if os.path.exists(args.sqlite):
            from_sqlite(generate=False)
        
        # initialize dataset
        voc_data = VOCDataset(args.data_dir, caption_support=True)
        unique_captions = list(
            set(list(map(lambda c: DatasetWrap.clean(c.captions), voc_data.train.caption.fetch(bulk=False)))))
        # add additional captions
        unique_captions.extend(list(map(DatasetWrap.clean, additional_captions)))
        unique_captions = list(filter(lambda txt: not emb_db.is_available(txt), unique_captions))
        Data.total_sentence = len(unique_captions)
        Data.total_tokens = sum(map(lambda x: len(x), unique_captions))
        print("Unique captions cleaned:", Data.total_sentence)
        print("Total tokens:", Data.total_tokens)
        return unique_captions
    
    bulk_embedded = False
    rpm = 3
    caption_loader = caption_loader(test_captions)
    bulk_caption_loader = OpenAITextLoader(caption_loader, Data.total_tokens, Data.total_sentence,
                                           rpm=rpm,
                                           tpm=150000, auto_sleep=False)
    cred_man = OpenAICredentialManager("./data/openai.apikey")
    cm = iter(cred_man)
    key, nickname = next(cm)
    for caption in tqdm(caption_loader):
        while True:
            model = OpenAIEmbeddings(openai_api_key=key, model="ada", max_retries=1)
            try:
                if cred_man.is_limit_exhausted(nickname):
                    raise RateLimitError("Rate limit exhausted for {}".format(nickname))
                
                # single
                embedding = model.embed_query(caption)
                emb_db.append(caption, embedding)
                emb_db.commit()
                # time.sleep(60 / rpm)
                break
            except RateLimitError:
                cred_man.set_limit_exhausted(nickname)
                key, nickname = next(cm)
    
    # For generating dataset
    if generate:
        Dataset.IMAGE_DIR = "JPEGImages"
        Dataset.CAPTION_DIR = "captions"
        # generate dataset
        generate_dataset(args)


def generate_caption_embedding_with_openai():
    args = parse_args()
    dataset = pathlib.Path(args.dataroot)
    # For reading images
    Dataset.IMAGE_DIR = "JPEGImages"
    # reading filepaths
    file_paths = list(list_dir(str(dataset / "train"), dir_flag=VOC_IMAGES, fullpath=True))
    file_paths.extend(list(list_dir(str(dataset / "test"), dir_flag=VOC_IMAGES, fullpath=True)))
    # generating dataset form SQLIte
    sqlite_data = SQLiteDataWrap(args.sqlite)
    sqlite_data.export_fast(args.data_dir, clean=args.clean, copy_images=args.copy_images, image_paths=file_paths)
    # For generating dataset
    Dataset.IMAGE_DIR = "JPEGImages"
    Dataset.CAPTION_DIR = "texts"
    # generate dataset
    generate_dataset(args)


def from_custom_dataset():
    args = parse_args()
    Dataset.IMAGE_DIR = "images"
    Dataset.ANNO_DIR = "Annotations"
    Dataset.CAPTION_DIR = "texts"
    # generate dataset
    generate_dataset(args)


def from_sqlite(generate=True):
    args = parse_args()
    dataset = pathlib.Path(args.dataroot)
    # For reading images
    Dataset.IMAGE_DIR = "JPEGImages"
    # reading filepaths
    file_paths = list(list_dir(str(dataset / "train"), dir_flag=VOC_IMAGES, fullpath=True))
    file_paths.extend(list(list_dir(str(dataset / "test"), dir_flag=VOC_IMAGES, fullpath=True)))
    # generating dataset form SQLIte
    sqlite_data = SQLiteDataWrap(args.sqlite)
    sqlite_data.export_fast(args.data_dir, clean=args.clean, copy_images=args.copy_images, image_paths=file_paths)
    
    if generate:
        # For generating dataset
        Dataset.IMAGE_DIR = "JPEGImages"
        Dataset.CAPTION_DIR = "texts"
        # generate dataset
        generate_dataset(args)


if __name__ == '__main__':
    # from_custom_dataset()
    create_openai_embedding_database(generate=True)
    # from_sqlite()

""" UBUNTU
sqlite3 -header -csv "C:\\Users\\dndlssardar\\Downloads\\tip_gai_22052023_1743.db" "SELECT * FROM caption" > caption.csv
python data/generate_custom_dataset.py --data_dir data/sixray_sample --emb_dim 300 --fasttext_model /mnt/c/Users/dndlssardar/Downloads/Fasttext/cc.en.300.bin
python data/generate_custom_dataset.py --data_dir data/sixray_500 --fasttext_model /data/fasttext/cc.en.300.bin  --sqlite /data/sixray_caption_db/<tip_gai.db> --clean --copy_images --dataroot /data/Sixray_easy/
python data/generate_custom_dataset.py --data_dir data/sixray_500 --fasttext_model /mnt/c/Users/dndlssardar/Downloads/Fasttext/cc.en.300.bin --sqlite data/tip_gai.db --clean --copy_images --dataroot "/mnt/c/Users/dndlssardar/OneDrive - Smiths Group/Documents/Projects/Dataset/Sixray_easy"
"""

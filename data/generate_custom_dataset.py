import argparse
import os.path
import pathlib
import pickle
from pprint import pprint

from voc_tools import reader as voc_reader
from voc_tools.utils import VOCDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--data_dir', dest='data_dir',
                        help='Path to dataset directory',
                        default='my_data', type=str, )
    parser.add_argument('--fasttext_vector', dest='fasttext_vector', type=str, default=None)
    # parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    # parser.add_argument('--manualSeed', type=int, help='manual seed', default=47)
    _args = parser.parse_args()
    pprint(_args)
    return _args


def generate_class_id_pickle(dataset_path):
    dataset_path = pathlib.Path(dataset_path)
    assert os.path.exists(str(dataset_path))

    pickle_class_info_path = str(dataset_path / "train" / "class_info.pickle")

    classes = VOCDataset(dataset_path).train.load().class_names()
    # save as pickle file
    with open(pickle_class_info_path, 'wb') as fpp:
        pickle.dump(classes, fpp)
        print("'{}' is created".format(pickle_class_info_path))


def generate_text_embedding_pickle(dataset_path, fasttext_vector):
    dataset_path = pathlib.Path(dataset_path)
    assert os.path.exists(str(dataset_path))

    classes = VOCDataset(dataset_path, caption_support=False).train.caption.fetch()


def generate_filename_pickle(dataset_path):
    dataset_path = pathlib.Path(dataset_path)
    assert os.path.exists(str(dataset_path))

    pickle_filenames_path = str(dataset_path / "train" / "filenames.pickle")

    # read the filenames as list and clean it
    mylist = list(voc_reader.list_dir(str(dataset_path / "train")))
    # save as pickle file
    with open(pickle_filenames_path, 'wb') as fpp:
        pickle.dump(mylist, fpp)
        print("'{}' is created".format(pickle_filenames_path))


if __name__ == '__main__':
    args = parse_args()
    generate_filename_pickle(args.data_dir)
    generate_class_id_pickle(args.data_dir)
    # generate_text_embedding_pickle(args.data_dir, args.fasttext_vector)

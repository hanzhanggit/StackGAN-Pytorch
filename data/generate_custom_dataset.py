import argparse
import os.path
import pathlib
import pickle
from pprint import pprint

from voc_tools.constants import VOC_IMAGES
from voc_tools.reader import list_dir
from voc_tools.utils import VOCDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--data_dir', dest='data_dir',
                        help='Path to dataset directory',
                        default='my_data', type=str, )
    parser.add_argument('--fasttext_vector', dest='fasttext_vector', type=str, default=None)
    parser.add_argument('--fasttext_model', dest='fasttext_model', type=str, default=None)
    # parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    # parser.add_argument('--manualSeed', type=int, help='manual seed', default=47)
    _args = parser.parse_args()
    pprint(_args)
    return _args


def generate_class_id_pickle(dataset_path):
    dataset_path = pathlib.Path(dataset_path)
    assert os.path.exists(str(dataset_path))

    classes = [annotations[0].class_name for annotations, image in VOCDataset(dataset_path).train.fetch(bulk=True)]
    # save as pickle file
    pickle_class_info_path = str(dataset_path / "train" / "class_info.pickle")
    with open(pickle_class_info_path, 'wb') as fpp:
        pickle.dump(classes, fpp)
        print("'{}' is created".format(pickle_class_info_path))


def generate_text_embedding_pickle(dataset_path, fasttext_vector=None, fasttext_model=None, embed_dim=100):
    """
    Generate and save caption embedding into a pickle file
    Args:
        dataset_path: VOC dataset path
        fasttext_vector: Pretrained fasttext vector (*.vec) file path See: https://fasttext.cc/docs/en/english-vectors.html
        fasttext_model: Pretrained fasttext  model (*.bin) file path See: https://fasttext.cc/docs/en/crawl-vectors.html
        embed_dim: Final embedding dimension
    """
    import fasttext
    import fasttext.util
    dataset_path = pathlib.Path(dataset_path)
    assert os.path.exists(str(dataset_path))
    model = None
    model_name = None
    if os.path.isfile(fasttext_model):
        model_name = pathlib.Path(fasttext_model).name
        print("Loading fasttext model:{}...".format(fasttext_model), end="")
        model = fasttext.load_model(fasttext_model)
        print("Loaded")
        if embed_dim != model.get_dimension():
            fasttext.util.reduce_model(model, embed_dim)
    assert model is not None, "A fasttext  model has to be initialised"
    print("Generating embeddings...", end="")
    embeddings = [list(map(lambda cap: model.get_word_vector(cap.captions), caption_list)) for caption_list in
                  VOCDataset(dataset_path, caption_support=True).train.caption.fetch(bulk=True)]
    print("Done.")
    print("Saving...")
    # save as pickle file
    pickle_path = str(dataset_path / "train" / "embeddings_{}_{}.pickle".format(model_name, embed_dim))
    with open(pickle_path, 'wb') as fpp:
        pickle.dump(embeddings, fpp)
        print("'{}' is created with {} entries".format(pickle_path, len(embeddings)))


def generate_filename_pickle(dataset_path):
    """
    Generate a list of 'filenames' per 'caption', then save it as pickle format.
    Note: This is not a unique filename list. If, for a certain image, we have multiple
    captions, then multiple entries with same 'filename' will be created.
    """
    dataset_path = pathlib.Path(dataset_path)
    assert os.path.exists(str(dataset_path))

    pickle_filenames_path = str(dataset_path / "train" / "filenames.pickle")

    # read the filenames from captions
    # mylist = [caption.filename.replace(".txt", ".jpeg") for caption in
    #           VOCDataset(dataset_path, caption_support=True).train.caption.fetch()]

    mylist = [os.path.basename(file) for file in list_dir(str(dataset_path / "train"), dir_flag=VOC_IMAGES)]

    # save as pickle file
    with open(pickle_filenames_path, 'wb') as fpp:
        pickle.dump(mylist, fpp)
        print("'{}' is created with {} entries".format(pickle_filenames_path, len(mylist)))


""" UBUNTU
python data/generate_custom_dataset.py --data_dir data/sixray_sample --fasttext_model /mnt/c/Users/dndlssardar/Downloads/Fasttext/cc.en.300.bin
"""
if __name__ == '__main__':
    args = parse_args()
    generate_filename_pickle(args.data_dir)
    generate_class_id_pickle(args.data_dir)
    # generate_text_embedding_pickle(args.data_dir, args.fasttext_vector, args.fasttext_model)

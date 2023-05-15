import argparse
import os.path
import pathlib
import pickle
from pprint import pprint

from voc_tools.constants import VOC_IMAGES
from voc_tools.reader import list_dir, from_file
from voc_tools.utils import VOCDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--data_dir', dest='data_dir',
                        help='Path to dataset directory',
                        default='my_data', type=str, )
    parser.add_argument('--fasttext_vector', dest='fasttext_vector', type=str, default=None)
    parser.add_argument('--fasttext_model', dest='fasttext_model', type=str, default=None)
    parser.add_argument('--emb_dim', dest='emb_dim', type=int, default=300)
    parser.add_argument('--bulk', dest='bulk', default=False, action="store_true")
    # parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    # parser.add_argument('--manualSeed', type=int, help='manual seed', default=47)
    _args = parser.parse_args()
    pprint(_args)
    return _args


def generate_class_id_pickle(dataset_path, bulk=False):
    """
    Generate and save CLASS_IDS into a pickle file.

        Each image typically contains one or many captions. In two ways we can save the 'class_ids'
    into the pickle file, 'class_ids' per caption and 'class_ids' per image. 'bulk' parameter
    controls this action.

    Args:
        dataset_path: VOC dataset path
        bulk: set `True` for 'class_ids per image' or set `False` 'class_ids per caption'
    """

    dataset_path = pathlib.Path(dataset_path)
    assert os.path.exists(str(dataset_path))

    if bulk:
        # 'class_ids per image'
        classes = [annotations[0].class_name for annotations, jpeg in VOCDataset(dataset_path).train.fetch(bulk=True)]
    else:
        # 'class_ids per caption'
        classes = list(
            map(lambda caption: list(from_file(str(dataset_path / "train" / "text" / caption.filename)))[0].class_name,
                VOCDataset(dataset_path, caption_support=True).train.caption.fetch(bulk=False)))
    unique = list(set(classes))
    class_id = list(map(lambda x: unique.index(x) + 1, classes))
    # save as pickle file
    pickle_class_info_path = str(dataset_path / "train" / "class_info.pickle")
    with open(pickle_class_info_path, 'wb') as fpp:
        pickle.dump(class_id, fpp)
        print("'{}' is created with {} entries".format(pickle_class_info_path, len(class_id)))


def generate_text_embedding_pickle(dataset_path, fasttext_vector=None, fasttext_model=None, emb_dim=300, bulk=False):
    """
    Generate and save caption embedding into a pickle file.

        Each image typically contains one or many captions. In two ways we can save the caption
    embeddings into the pickle file. In as single instance or in bulk. Let, say we have 3 captions per image, and we
    have 2 images. If we want to store the embeddings as bulk, then it should look like:
    [[emb11, emb12, emb13], [emb21, emb22, emb23]]. If we want to store the embeddings as single instance, then it will be:
    [[emb11], [emb12], [emb13], [emb21], [emb22], [emb23]].

    Args:
        dataset_path: VOC dataset path
        fasttext_vector: Pretrained fasttext vector (*.vec) file path See: https://fasttext.cc/docs/en/english-vectors.html
        fasttext_model: Pretrained fasttext  model (*.bin) file path See: https://fasttext.cc/docs/en/crawl-vectors.html
        emb_dim: Final embedding dimension
        bulk: This boolean flag specifies whether to store as bulk or single
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
        if emb_dim != model.get_dimension():
            fasttext.util.reduce_model(model, emb_dim)
    assert model is not None, "A fasttext  model has to be initialised"
    print("Generating embeddings...", end="")
    if bulk:
        embeddings = [list(map(lambda cap: model.get_word_vector(cap.captions), caption_list)) for caption_list in
                      VOCDataset(dataset_path, caption_support=True).train.caption.fetch(bulk=True)]
    else:
        embeddings = [[model.get_word_vector(caption.captions)] for caption in
                      VOCDataset(dataset_path, caption_support=True).train.caption.fetch(bulk=False)]
    print("Done.")
    print("Saving...")
    # save as pickle file
    pickle_path = str(dataset_path / "train" / "embeddings_{}_{}D.pickle".format(model_name, emb_dim))
    with open(pickle_path, 'wb') as fpp:
        pickle.dump(embeddings, fpp)
        print("'{}' is created with {} entries".format(pickle_path, len(embeddings)))


def generate_filename_pickle(dataset_path, bulk=False):
    """
    Generate a list of 'filenames' per 'caption', then save it as pickle format.
    Note: This is not a unique filename list. If, for a certain image, we have multiple
    captions, then multiple entries with same 'filename' will be created.

        Each image typically contains one or many captions. In two ways we can save the filenames
    into the pickle file, filename per caption and filename per image. 'bulk' parameter
    controls this action.

    Args:
        dataset_path: the path to the dataset
        bulk: set `True` for 'filename per image' or set `False` 'filename per caption'
    """
    dataset_path = pathlib.Path(dataset_path)
    assert os.path.exists(str(dataset_path))

    pickle_filenames_path = str(dataset_path / "train" / "filenames.pickle")

    if bulk:
        mylist = [os.path.basename(file) for file in list_dir(str(dataset_path / "train"), dir_flag=VOC_IMAGES)]
    else:
        # read the filenames from captions
        mylist = [caption.filename.replace(".txt", ".jpg") for caption in
                  VOCDataset(dataset_path, caption_support=True).train.caption.fetch(bulk=False)]
    mylist = list(map(lambda x: os.path.join("train", "JPEGImages", x), mylist))
    # save as pickle file
    with open(pickle_filenames_path, 'wb') as fpp:
        pickle.dump(mylist, fpp)
        print("'{}' is created with {} entries".format(pickle_filenames_path, len(mylist)))


""" UBUNTU
python data/generate_custom_dataset.py --data_dir data/sixray_sample --fasttext_model /mnt/c/Users/dndlssardar/Downloads/Fasttext/cc.en.300.bin --emb_dim 300
"""
if __name__ == '__main__':
    args = parse_args()
    generate_filename_pickle(args.data_dir, args.bulk)
    generate_class_id_pickle(args.data_dir, args.bulk)
    generate_text_embedding_pickle(args.data_dir, args.fasttext_vector, args.fasttext_model, emb_dim=args.emb_dim,
                                   bulk=args.bulk)

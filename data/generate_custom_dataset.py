import argparse
import os.path
import pathlib
import pickle
import shutil
from collections import defaultdict
from pprint import pprint

from voc_tools.constants import VOC_IMAGES
from voc_tools.reader import list_dir, from_file
from voc_tools.utils import Dataset, VOCDataset
import sqlite3


def generate_class_id_pickle(dataset_path, classes):
    """
    Generate and save CLASS_IDS into a pickle file

    Args:
        dataset_path: VOC dataset path
        classes: a list of classes to add to dataset
    """
    dataset_path = pathlib.Path(dataset_path)
    assert os.path.exists(str(dataset_path))
    # categorical encoding of the classes
    unique = list(set(classes))
    class_id = list(map(lambda x: unique.index(x) + 1, classes))
    # save as pickle file
    pickle_class_info_path = str(dataset_path / "train" / "class_info.pickle")
    with open(pickle_class_info_path, 'wb') as fpp:
        pickle.dump(class_id, fpp)
        print("'{}' is created with {} entries".format(pickle_class_info_path, len(class_id)))


def get_embedding_model(fasttext_vector=None, fasttext_model=None, emb_dim=300):
    """
    Args:
        fasttext_vector: Pretrained fasttext vector (*.vec) file path See: https://fasttext.cc/docs/en/english-vectors.html
        fasttext_model: Pretrained fasttext  model (*.bin) file path See: https://fasttext.cc/docs/en/crawl-vectors.html
        emb_dim: Final embedding dimension
    """
    import fasttext
    import fasttext.util
    model = None
    model_name = ""
    if os.path.isfile(fasttext_model):
        model_name = pathlib.Path(fasttext_model).name
        print("Loading fasttext model:{}...".format(fasttext_model), end="")
        model = fasttext.load_model(fasttext_model)
        print("Loaded")
        if emb_dim != model.get_dimension():
            fasttext.util.reduce_model(model, emb_dim)
    assert model is not None, "A fasttext  model has to be initialised"
    return model, model_name, emb_dim


def generate_text_embedding_pickle(dataset_path, embeddings, model_name, emb_dim):
    """
    Generate and save caption embedding into a pickle file.
    Args:
        dataset_path: VOC dataset path
        embeddings: Prepared caption embeddings
        model_name: Model name which is used to prepare the embeddings
        emb_dim: Final embedding dimension
    """
    dataset_path = pathlib.Path(dataset_path)
    assert os.path.exists(str(dataset_path))
    # save as pickle file
    pickle_path = str(dataset_path / "train" / "embeddings_{}_{}D.pickle".format(model_name, emb_dim))
    with open(pickle_path, 'wb') as fpp:
        pickle.dump(embeddings, fpp)
        print("'{}' is created with {} entries".format(pickle_path, len(embeddings)))


def generate_filename_pickle(dataset_path, filenames):
    """
    Generate a list of 'filenames' per 'caption', then save it as pickle format.
    Note: This is not a unique filename list. If, for a certain image, we have multiple
    captions, then multiple entries with same 'filename' will be created.

    Args:
        dataset_path: the path to the dataset
    """
    dataset_path = pathlib.Path(dataset_path)
    assert os.path.exists(str(dataset_path))

    pickle_filenames_path = str(dataset_path / "train" / "filenames.pickle")

    # save as pickle file
    with open(pickle_filenames_path, 'wb') as fpp:
        pickle.dump(filenames, fpp)
        print("'{}' is created with {} entries".format(pickle_filenames_path, len(filenames)))


class DatasetWrap:
    def __init__(self, dataset_path, bulk=False, class_ids=False) -> None:
        dataset_path = pathlib.Path(dataset_path)
        assert os.path.exists(str(dataset_path))
        self.dataset_path = pathlib.Path(dataset_path)
        self.is_bulk = bulk
        self.class_ids = class_ids
        self.voc_data = VOCDataset(dataset_path, caption_support=True)

    def _prepare_classes(self):
        """
            Each image typically contains one or many captions. In two ways we can save the 'class_ids'
        into the pickle file, 'class_ids' per caption and 'class_ids' per image. 'bulk' parameter
        controls this action.
        """
        dataset_path = self.dataset_path
        if self.is_bulk:
            # 'class_ids per image'
            self.classes = [annotations[0].class_name for annotations, jpeg in self.voc_data.train.fetch(bulk=True)]
        else:
            # 'class_ids per caption'
            self.classes = list(
                map(lambda caption:
                    list(from_file(str(dataset_path / "train" / Dataset.CAPTION_DIR / caption.filename)))[
                        0].class_name,
                    self.voc_data.train.caption.fetch(bulk=False)))

    def _prepare_filenames(self):
        """
            Each image typically contains one or many captions. In two ways we can save the filenames
        into the pickle file, filename per caption and filename per image. 'bulk' parameter
        controls this action.
        """
        dataset_path = self.dataset_path

        if self.is_bulk:

            mylist = [os.path.basename(file) for file in list_dir(str(dataset_path / "train"), dir_flag=VOC_IMAGES)]
        else:
            # read the filenames from captions
            mylist = [caption.filename.replace(".txt", ".jpg") for caption in
                      self.voc_data.train.caption.fetch(bulk=False)]
        self.filenames = list(map(lambda x: os.path.join("train", Dataset.IMAGE_DIR, x), mylist))  # fill path names

    def _prepare_embeddings(self, model):
        """
            Each image typically contains one or many captions. In two ways we can save the caption
        embeddings into the pickle file. In as single instance or in bulk. Let, say we have 3 captions per image, and we
        have 2 images. If we want to store the embeddings as bulk, then it should look like:
        [[emb11, emb12, emb13], [emb21, emb22, emb23]]. If we want to store the embeddings as single instance, then it will be:
        [[emb11], [emb12], [emb13], [emb21], [emb22], [emb23]].
        """
        dataset_path = self.dataset_path
        if self.is_bulk:
            self.embeddings = [list(map(lambda cap: model.get_word_vector(cap.captions), caption_list)) for caption_list
                               in
                               VOCDataset(dataset_path, caption_support=True).train.caption.fetch(bulk=True)]
        else:
            self.embeddings = [[model.get_word_vector(caption.captions)] for caption in
                               VOCDataset(dataset_path, caption_support=True).train.caption.fetch(bulk=False)]

    def prepare_dataset(self, model, model_name, emb_dim):
        if self.class_ids:
            self._prepare_classes()
        self._prepare_filenames()
        self._prepare_embeddings(model)
        generate_filename_pickle(str(self.dataset_path), self.filenames)
        if self.class_ids:
            generate_class_id_pickle(str(self.dataset_path), self.classes)
        generate_text_embedding_pickle(str(self.dataset_path), self.embeddings, model_name, emb_dim)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--data_dir', dest='data_dir',
                        help='Path to dataset directory.',
                        default='my_data', type=str, )
    parser.add_argument('--bulk', dest='bulk', default=False, action="store_true")
    parser.add_argument('--class_id', dest='class_id', default=False, action="store_true")
    parser.add_argument('--fasttext_vector', dest='fasttext_vector', type=str, default=None)
    parser.add_argument('--fasttext_model', dest='fasttext_model', type=str, default=None)
    parser.add_argument('--emb_dim', dest='emb_dim', type=int, default=300)
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
    # parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    # parser.add_argument('--manualSeed', type=int, help='manual seed', default=47)
    _args = parser.parse_args()
    pprint(_args)
    return _args


def from_custom_dataset():
    args = parse_args()
    Dataset.IMAGE_DIR = "images"
    Dataset.ANNO_DIR = "Annotations"
    Dataset.CAPTION_DIR = "texts"
    emb_model, emb_model_name, emb_dimension = get_embedding_model(args.fasttext_vector, args.fasttext_model,
                                                                   emb_dim=args.emb_dim)
    vdw = DatasetWrap(args.data_dir, args.bulk, args.class_id)
    vdw.prepare_dataset(emb_model, emb_model_name, emb_dimension)


class Caption:
    def __init__(self, items):
        self.idx = items[0]
        self.file_id = items[1]
        self.filename = self.file_id.replace(".jpg", ".txt")
        self.caption = items[2]
        self.author = items[3]
        self.is_occluded = items[4]
        self.is_error = items[5]


def check_grammar(text):
    # Create a LanguageTool instance
    try:
        from language_tool_python import LanguageTool
        tool = LanguageTool('en-US')

        # Check grammar in the text
        matches = tool.check(text)
        return matches
    except:
        return []


class SQLiteDataWrap:

    def __init__(self, dbpath):
        assert os.path.isfile(dbpath), dbpath
        conn = sqlite3.connect(dbpath)
        self.conn = conn
        self.dbpath = dbpath
        self.is_clean = False
        self.image_path_dict = None

    def clean(self):
        self.is_clean = True
        return self

    def get_path(self, file_id, image_paths):
        if self.image_path_dict is None:
            self.image_path_dict = defaultdict(lambda: None)
        filepath = self.image_path_dict[file_id]
        if filepath is None:
            filtered = list(filter(lambda fname: file_id in fname, image_paths))
            if len(filtered) > 0:
                filepath = filtered[0]
                self.image_path_dict[file_id] = filepath
        return filepath

    def export(self, data_root, clean=False, copy_images=False, image_paths=()):
        data_root = pathlib.Path(data_root)

        # defining paths
        caption_root = data_root / "train" / "texts"
        image_root = data_root / "train" / "JPEGImages"
        if clean:
            print("Cleaning previous data...", end="")
            # deleting directories
            shutil.rmtree(caption_root, ignore_errors=True)
            shutil.rmtree(image_root, ignore_errors=True)
            print("Done")
        # create directories
        os.makedirs(caption_root, exist_ok=True)
        os.makedirs(image_root, exist_ok=True)
        os.makedirs(data_root / "test" / "JPEGImages", exist_ok=True)
        os.makedirs(data_root / "test" / "JPEGImages", exist_ok=True)

        statistics = defaultdict(lambda: defaultdict(lambda: 0))
        # taking DB cursor and running queries
        print("Quering dataset form SQLITE...")
        curr = self.conn.cursor()
        dataset = curr.execute("SELECT * FROM caption")
        print("Creating dataset form SQLITE...", end="")
        for data in dataset:
            caption = Caption(data)
            if copy_images:
                # copying images
                filepath = self.get_path(caption.file_id, image_paths)
                if not os.path.isfile(image_root / caption.file_id):
                    shutil.copyfile(filepath, image_root / caption.file_id)

            # creating the caption files
            with open(caption_root / caption.filename, "a+") as fp:
                fp.seek(0)
                lines = fp.readlines()
                if caption.caption + "\n" in lines:
                    statistics[caption.filename]['duplicate'] += 1
                    statistics["caption"]['duplicate'] += 1
                elif len(caption.caption.split(" ")) < 5:
                    statistics[caption.filename]['faulty'] += 1
                    statistics["caption"]['faulty'] += 1
                else:
                    fp.write(caption.caption + "\n")
                    # check grammar
                    errors = check_grammar(caption.caption)
                    # save statistics
                    statistics[caption.filename]['caption'] += 1
                    statistics[caption.filename]['grammar'] += len(errors) > 0
                    statistics["caption"]['total'] += 1
                    statistics['grammar']['error-sentence'] += len(errors) > 0
                    statistics['grammar']['total-error'] += len(errors)
        print("Done")
        return statistics


def from_sqlite():
    args = parse_args()
    dataset = pathlib.Path(args.dataroot)
    # For reading images
    Dataset.IMAGE_DIR = "JPEGImages"
    # reading filepaths
    file_paths = list(list_dir(str(dataset / "train"), dir_flag=VOC_IMAGES, fullpath=True))
    file_paths.extend(list(list_dir(str(dataset / "test"), dir_flag=VOC_IMAGES, fullpath=True)))
    # generating dataset form SQLIte
    # sqlite_data = SQLiteDataWrap(args.sqlite)
    # sqlite_data.clean().export(args.data_dir, clean=args.clean, copy_images=args.copy_images, image_paths=file_paths)
    # For generating dataset
    Dataset.IMAGE_DIR = "JPEGImages"
    Dataset.CAPTION_DIR = "texts"
    # loading embedding
    emb_model, emb_model_name, emb_dimension = get_embedding_model(args.fasttext_vector, args.fasttext_model,
                                                                   emb_dim=args.emb_dim)
    # creating object to generate compatible pickle files for StackGAN
    vdw = DatasetWrap(args.data_dir, args.bulk, args.class_id)
    vdw.prepare_dataset(emb_model, emb_model_name, emb_dimension)


if __name__ == '__main__':
    # from_custom_dataset()
    from_sqlite()

""" UBUNTU
python data/generate_custom_dataset.py --data_dir data/sixray_sample --emb_dim 300 --fasttext_model /mnt/c/Users/dndlssardar/Downloads/Fasttext/cc.en.300.bin
python data/generate_custom_dataset.py --data_dir data/sixray_500 --fasttext_model /mnt/c/Users/dndlssardar/Downloads/Fasttext/cc.en.300.bin --sqlite data/tip_gai.db --clean --copy_images --dataroot "/mnt/c/Users/dndlssardar/OneDrive - Smiths Group/Documents/Projects/Dataset/Sixray_easy"
"""
